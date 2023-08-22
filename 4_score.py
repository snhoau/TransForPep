from pycpd import RigidRegistration
import numpy as np
import pandas as pd
import pickle, time


def normzero(data):
    x_mean_target = data[:, 0].mean()
    y_mean_target = data[:, 1].mean()
    z_mean_target = data[:, 2].mean()

    return data - [x_mean_target, y_mean_target, z_mean_target]


def eculidDisSim(x, y):
    '''
    欧几里得相似度
    '''
    return np.sqrt(sum(pow(a - b, 2) for a, b in zip(x, y)))


def list2array(x):
    dff = pd.concat([pd.DataFrame({'{}'.format(index): labels}) for index, labels in enumerate(x)], axis=1)
    return dff.fillna(0).values.T.astype(float)


def compE(x, y):
    if x > 0 and y < 0:
        return -1
    elif x < 0 and y > 0:
        return -1
    else:
        return abs(abs(x) - abs(y))


def compM(x, y):
    if x == y:
        return 1
    else:
        return -1


if __name__ == '__main__':
    # load data and score
    with open('dis_bchhh_cbem350.pkl', "rb") as fs:
        dis_all = pickle.load(fs)
    with open('e_bchhh_cbem350.pkl', "rb") as fs:
        e_all = pickle.load(fs)
    with open('m_bchhh_cbem350.pkl', "rb") as fs:
        m_all = pickle.load(fs)

    image_target_list = []
    with open('search\Database\map_bchhh.txt', 'r') as fp:
        content = fp.readlines()
    str_list = [s.rstrip().split() for s in content]
    image_target_list = [x for x in str_list]


    # screen the top candidate
    # dis_sort=sorted(range(len(dis_all)), key=dis_all.__getitem__)#min batter
    e_score = []

    for i in e_all:
        ne = 0
        es = []
        for xz in i:
            if xz == -1:
                ne += 1
            else:
                es.append(xz)
        e_score.append(sum(es)/len(es)+ne/len(i))  # min batter

    m_score = []
    for i in m_all:
        nm=0
        mm=0
        for xz in i:
            if xz == -1:
                nm+=1
            else:
                mm+=1
        m_score.append(mm/len(i))  

    score = list(np.array(m_score)-((np.array(dis_all) + np.array(e_score)))) # max batter
    s_sort = sorted(range(len(score)), key=score.__getitem__)
    print(s_sort[-10:])
    # to file
    fm = []
    for i in s_sort:
        fm.append([image_target_list[i-1], score[i], e_score[i], m_score[i]])
    fx=open('file_sort_score.txt', 'w')
    for line in fm:
        fx.write(str(line)+'\n')
    fx.close()

