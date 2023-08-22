# -*- coding: utf-8 -*-
# These codes are adapted from Transformer and I have already cited the source.
# If you want to use them again, please cite the original work.  arXiv:1706.03762v5
# @Author  : Wang Gan
# @Email   : wang.gan@outlook.com


import copy
import math
import numpy
import os
import pickle
import time
import warnings
from os.path import exists

import GPUtil
import altair as alt
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.functional import log_softmax
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split

# Set to False to skip notebook execution (e.g. for debugging)
warnings.filterwarnings("ignore")
RUN_EXAMPLES = True
autocast = torch.cuda.amp.autocast


class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        # return log_softmax(self.proj(x), dim=-1)
        xeg = x[:, :, 3:]
        xegr = log_softmax(self.proj(xeg), dim=-1)
        return torch.cat((x[:, :, :3], xegr), dim=2)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
                .contiguous()
                .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        # return self.lut(x) * math.sqrt(self.d_model)
        xe = x[:, :, -1].long()
        xeb = self.lut(xe) * math.sqrt(self.d_model)
        return torch.cat((x[:, :, :-1], xeb), dim=2)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


def make_model(
        src_vocab, tgt_vocab, N=6, d_model=32, d_ff=64, h=8, dropout=0.1
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model-3, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model-3, tgt_vocab), c(position)),
        Generator(d_model-3, tgt_vocab),
    )  # for protein Generator\Embeddings, d_model-4

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, tgt=None, pad=0):  # 2 = <blank>, 0 for protein
        self.src = src
        # self.src_mask = (src != pad).unsqueeze(-2)
        self.src_mask = (src[:, :, -1] != pad).unsqueeze(-2)
        if tgt is not None:
            # self.tgt = tgt[:, :-1]
            self.tgt = tgt[:, :-1, :]
            # self.tgt_y = tgt[:, 1:]
            self.tgt_y = tgt[:, 1:, :]# from 1000 to 999
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            # self.ntokens = (self.tgt_y != pad).data.sum()
            self.ntokens = (self.tgt_y[:, :, -1] != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        # tgt_mask = (tgt != pad).unsqueeze(-2)
        # tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        tgt_mask = (tgt[:, :, -1] != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-2)).type_as(tgt_mask.data)
        return tgt_mask


class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed


def run_epoch(
        data_iter,
        model,
        loss_compute,
        optimizer,
        scheduler,
        mode="train",
        accum_iter=1,
        train_state=TrainState(),
):
    """Train a single epoch"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for i, batch in enumerate(data_iter):
        #with autocast():
        out = model.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)  # batch.ntokens = norm
        # loss_node = loss_node / accum_iter
        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 80 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                        "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                        + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens, train_state


def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
            model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.long().data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())



def loss(x, crit):
    d = x + 3 * 1
    predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d]])
    return crit(predict.log(), torch.LongTensor([1])).data


class SimpleLossCompute:
    "A simple loss compute and train function."

    # def __init__(self, generator, criterion1, criterion2):
    def __init__(self, generator, criterion1, criterion2):
        self.generator = generator
        # self.criterion = criterion
        self.criterion1 = criterion1
        self.criterion2 = criterion2

    def __call__(self, x, y, norm):
        x = self.generator(x)
        # sloss = (self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm)
        sloss = (self.criterion1(x[:, :, 3:].contiguous().view(-1, x.size(-1) - 3),
                                 y[:, :, 3:].contiguous().view(-1)) / norm)
        loss2 = self.criterion2(x[:, :, :3].contiguous().view(-1, x.size(-1) - 631),
                                y[:, :, :3].contiguous().view(-1, y.size(-1) - 1))
        # return sloss.data * norm, sloss
        return (sloss.data + loss2.data) * norm, sloss + loss2


class MyDataset_f(Dataset):

    def __init__(self, dir, device):
        self.data_dir = dir
        self.device = device

        self.image_target_list = []

        with open(os.path.join(dir, 'map.txt'), 'r') as fp:
            content = fp.readlines()

            str_list = [s.rstrip().split() for s in content]
            self.image_target_list = [x for x in str_list]

    def __getitem__(self, index):
        image_label_pair = self.image_target_list[index]
        #print(image_label_pair[0])
        with open(image_label_pair[0], 'rb') as f:
            datainfo = pickle.load(f)

        return torch.tensor(numpy.array(datainfo), dtype=torch.float32, device=self.device)

    def __len__(self):
        return len(self.image_target_list)


def create_dataloaders(
        device,
        txtpath,
        batch_size=2000,
        tv_rate=0.97,
        is_distributed=False,
):
    # the source data of interface, with x,y,z,[e,m]
    data = MyDataset_f(txtpath, device=device)
    train_size = round(tv_rate * len(data))
    train_dataset, valid_dataset = random_split(
        dataset=data,
        lengths=[train_size, len(data) - train_size],
        generator=torch.Generator().manual_seed(0)
    )
    print('size of train dataset -', len(train_dataset))
    print('size of valid dataset -', len(valid_dataset))

    # do suffle and batch_size
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    return train_dataloader, valid_dataloader


def train_worker(
        gpu,
        ngpus_per_node,
        vocab_src,
        vocab_tgt,
        # spacy_de,
        # spacy_en,
        config,
        is_distributed=False,
        test_flag=True
):
    print(f"Train worker process using GPU: {gpu} for training", flush=True)
    torch.cuda.set_device(gpu)

    # pad_idx = vocab_tgt["<blank>"]
    pad_idx = vocab_tgt[0]
    d_model = 32

    model = make_model(len(vocab_src), len(vocab_tgt), N=24)
    if test_flag:# load saved model
        print("loading model...")
        model.load_state_dict(torch.load("transpep_model_dis350_cbem_ffff.pt"))
    model.cuda(gpu)
    module = model
    is_main_process = True
    if is_distributed:
        dist.init_process_group(
            "nccl", init_method="env://", rank=gpu, world_size=ngpus_per_node
        )
        model = DDP(model, device_ids=[gpu])
        module = model.module
        is_main_process = gpu == 0

    # criterion = LabelSmoothing(
    #    size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1
    # )
    # criterion.cuda(gpu)
    criterion1 = LabelSmoothing(size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1)
    criterion1.cuda(gpu)
    criterion2 = nn.SmoothL1Loss()
    criterion2.cuda(gpu)

    # train_dataloader, valid_dataloader = create_dataloaders(
    #    gpu,
    #    vocab_src,
    #    vocab_tgt,
    #    spacy_de,
    #    spacy_en,
    #    batch_size=config["batch_size"] // ngpus_per_node,
    #    max_padding=config["max_padding"],
    #    is_distributed=is_distributed,
    # )
    train_dataloader, valid_dataloader = create_dataloaders(
        device=device,
        txtpath="Dataset_dis350_cbem\\",
        batch_size=config["batch_size"],
        tv_rate=0.97,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, d_model, factor=1, warmup=config["warmup"]
        ),
    )
    train_state = TrainState()

    for epoch in range(config["num_epochs"]):
        if is_distributed:
            train_dataloader.sampler.set_epoch(epoch)
            valid_dataloader.sampler.set_epoch(epoch)

        model.train()
        print(f"[GPU{gpu}] Epoch {epoch} Training ====", flush=True)

        _, train_state = run_epoch(
            (Batch(b.transpose(0, 1)[0], b.transpose(0, 1)[1], pad_idx) for b in train_dataloader),
            # Batch(b[0], b[1], pad_idx) for b in train_dataloader),
            model,
            SimpleLossCompute(model.generator, criterion1, criterion2),
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=config["accum_iter"],
            train_state=train_state,
        )
        # SimpleLossCompute(module.generator, criterion)
        GPUtil.showUtilization()
        if is_main_process:
            file_path = "%s%.2d.pt" % (config["file_prefix"], epoch)
            torch.save(module.state_dict(), file_path)
        torch.cuda.empty_cache()

        print(f"[GPU{gpu}] Epoch {epoch} Validation ====", flush=True)
        model.eval()
        sloss = run_epoch(
            (Batch(b.transpose(0, 1)[0], b.transpose(0, 1)[1], pad_idx) for b in valid_dataloader),
            model,
            SimpleLossCompute(model.generator, criterion1, criterion2),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )
        # SimpleLossCompute(module.generator, criterion)
        print(sloss)
        torch.cuda.empty_cache()

    if is_main_process:
        file_path = "%sfinal.pt" % config["file_prefix"]
        torch.save(module.state_dict(), file_path)


def load_trained_model(test_flag=True):
    config = {
        "batch_size": 32,
        "distributed": False,
        "num_epochs": 20000,
        "accum_iter": 10,
        "base_lr": 1.0,
        "max_padding": 350,
        "warmup": 3000,
        "file_prefix": "transpep_model_dis350_cbem_",
    }

    model_path = "transpep_model_dis350_cbem_final.pt"
    if not exists(model_path):

        vocab_src = vocab_tgt = list(range(631))#charge to load dic
        train_worker(0, 1, vocab_src, vocab_tgt, config, False, test_flag)
        # train_worker(0, 1, vocab_src, vocab_tgt, spacy_de, spacy_en, config, False)

    model = make_model(len(vocab_src), len(vocab_tgt), N=24)
    model.load_state_dict(torch.load("transpep_model_dis350_cbem_final.pt"))
    return model



if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(test_flag=False)# if true than continue train model
