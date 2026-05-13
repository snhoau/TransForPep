# -*- coding: utf-8 -*-
# @Author  : Wang Gan
# @Email   : snhoau@hotmail.com
import pickle

if __name__ == '__main__':
    #fatch the score
    '''
    sco=[]
    with open("file_sort_bchhh.txt") as file:
        for item in file:
            p1=item.split(',')
            sco.append([p1[0][51:63],p1[1]])
    # transfer to dic
    sco_d={}
    for i in sco:
        sco_d[i[0]]=i[1]
    with open('bchhh_score_dic.pkl', "wb") as fs:
        pickle.dump(sco_d, fs)
    
    '''
    with open('bchhh_score_dic.pkl', "rb") as fs:
        sco_d=pickle.load(fs)
    # transfer to dic
    print(len(sco_d))

    trg_scr=[]
    with open("seq2.blast") as file:
        for item in file:
            if item[0:7] == 'HHH_bc_':
                print(item[0:12] + ' ',sco_d[item[0:12]])
                trg_scr.append([item[0:12] + ' ',sco_d[item[0:12]]])
    with open('bchhh_blast_score.pkl', "wb") as fs:
        pickle.dump(trg_scr, fs)
    print('fasta get score')











