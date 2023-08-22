# -*- coding: utf-8 -*-
# These codes are adapted from Transformer and I have already cited the source.
# If you want to use them again, please cite the original work.  arXiv:1706.03762v5
# @Author  : Wang Gan
# @Email   : Wang.gan@outlook.com


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
        xeg = x[:, 3:]
        xegr = log_softmax(self.proj(xeg), dim=-1)
        return torch.cat((x[:, :3], xegr), dim=1)


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
        xe = x[:, -1].long()  # without batch , reduce 1 dim
        xeb = self.lut(xe) * math.sqrt(self.d_model)
        return torch.cat((x[:, :-1], xeb), dim=1)  # without batch , reduce 1 dim


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
        x = x + self.pe[:, : x.size(0)].requires_grad_(False)  # no batch
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
        nn.Sequential(Embeddings(d_model - 3, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model - 3, tgt_vocab), c(position)),
        Generator(d_model - 3, tgt_vocab),
    )  # for protein Generator\Embeddings, d_model-3

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def load_trained_model(test_flag=True):

    vocab_src = vocab_tgt = list(range(631))

    model = make_model(len(vocab_src), len(vocab_tgt), N=24)
    model.load_state_dict(torch.load("transpep_model_dis350_cbem_final.pt"))
    return model




if __name__ == '__main__':

    # gpu or not
    device = torch.device("cpu")
    metype = pickle.load(open('cbem.pkl', 'rb'))# load the e, m combination
    valu = list(range(len(metype)))

    vcbr = {0: 'H', 11: 'C', 14: 'N', 16: 'O', 18: 'S'}
    vcb = {0: 0, 1: 11, 2: 14, 3: 16, 4: 18}  # atmtype
    model = load_trained_model()  # if true continue train model
    model = model.eval()
    max_length = 350  # change to 1000 for all protein and with model

    with open('gsdmdinput.pkl', 'rb') as f:
        datainfo = pickle.load(f)
    a = numpy.zeros((max_length - len(datainfo), 5))
    a_l = numpy.vstack((numpy.array(datainfo), a))
    # get src
    src = []
    for i in list(a_l):
        try:
            src.append([i[0], i[1], i[2], metype.index([vcb[i[4]], i[3]])])  # if the cbem has in the metype

        except:
            mtm = []  # if not in metype, first make the m correct
            for z in metype:
                if z[0] == vcb[i[4]]:
                    mtm.append(metype.index(z))
            ebuf = []  # get the min dis e
            for w in mtm:
                if metype[w][1] * i[3] > 0:
                    ebuf.append(abs(metype[w][1]) - abs(i[3]))
                else:
                    ebuf.append(abs(metype[w][1] - i[3]))
            src.append([i[0], i[1], i[2], mtm[ebuf.index(min(ebuf))]])


    src = torch.tensor(src, dtype=torch.float32, device=device)
    tgt = src
    pad = 0
    src_mask = (src[:, -1] != pad).unsqueeze(-2)
    tgt = tgt[:-1, :]
    tgt_mask = (tgt[:, -1] != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & subsequent_mask(tgt.size(-2)).type_as(tgt_mask.data)
    tgt_suf = torch.tensor([], dtype=torch.float32, device=device)
    out = model(src, tgt, src_mask, tgt_mask)  # 1,349,32
    for i in range(max_length - 1):
        predict = model.generator(out[:, i])
        _, y = torch.max(predict[:, 3:], dim=1)  # 4-3 includ e
        tgt_suf = torch.concat([tgt_suf, predict[:, 0:3], y.unsqueeze(0)], dim=1)
        if y == 0:
            print('get 0')
            break
    print(tgt_suf.shape)
    print(tgt_suf.reshape(-1, 4))
    idsuf = tgt_suf.reshape(-1, 4).detach().numpy()

    # building xyz file
    em = []
    for i in list(idsuf):
        em.append([i[0], i[1], i[2], metype[int(i[3])][1], metype[int(i[3])][0]])  # x,y,z,[e,m]

    p1 = numpy.array(em)  # get e m from vocb
    numpy.save('idlsuf/idesuf_dis350_cbem', p1)
    p1 = p1[:, [4, 0, 1, 2]]  # remove e; m,x,y,z
    lindata = []
    idx = 0
    strx = '\n'
    fil = 'idlsuf/idesuf_dis350_cbem.xyz'
    for i in p1:
        p2 = [vcbr[i[0].tolist()], i[1].tolist(), i[2].tolist(), i[3].tolist()]
        lindata.append(p2)
        if idx == 0:
            with open(fil, 'a') as f:
                f.writelines(str(p1.shape[0]) + strx)
                f.writelines(str('idesud  0.000000') + strx)
        s = str(p2).replace('[', '').replace(']', '').replace('\'', '').replace(',', '')

        with open(fil, 'a') as f:
            f.writelines(s + strx)
        idx += 1
    print('conver to xyz... ')
