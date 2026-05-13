# -*- coding: utf-8 -*-
# @Author  : Wang Gan
# @Email   : snhoau@hotmail.com


if __name__ == '__main__':
    with open("bchhh_scaffolds.seq") as file:
        for item in file:
            p1=item.split(' ')
            with open('bchhh_scaffolds.fasta','a') as f:
                f.writelines('>'+str(p1[1].replace('\\n','')))
                f.writelines(str(p1[0]) +'\n')
    print('conver to fasta... ')











