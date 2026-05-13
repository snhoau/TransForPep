# -*- coding: utf-8 -*-
# @Author  : Wang Gan
# @Email   : snhoau@hotmail.com
import pickle

if __name__ == '__main__':
    #fatch the score
    
    sco=[]
    with open("file_sort_bchhh.txt") as file:
        for item in file:
            p1=item.split(',')
            sco.append([p1[0][51:63],p1[1]])
    print(len(sco))
    # get seq
    seque=[]
    with open('bchhh_scaffolds.fasta') as file:
        for item in file:
            if item[0] == '>':
                seque.append(item[1:-1])
            else:
                seque.append(item)
    seque2={}
    if (len(seque)%2 ==0):
        for idx in range(0,len(seque),2):
            seque2[seque[idx]]=seque[idx+1]
    else:
        print('out of range')
    #fatch seq

    idseqscr=[]
    for i in sco:
        idseqscr.append([i[0],seque2[i[0]]])
    # save to fasta
    for i in idseqscr:
        with open('bchhh_scaffolds_select.fasta','a') as f:
            f.writelines('>'+str(i[0]+'\n'))
            f.writelines(str(i[1]))
    print('save to fasta')













