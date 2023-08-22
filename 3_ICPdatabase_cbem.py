from pycpd import RigidRegistration
import numpy as np
import pandas as pd
import pickle,time


def normzero(data):
    x_mean_target = data[:, 0].mean()
    y_mean_target = data[:, 1].mean()
    z_mean_target = data[:, 2].mean()

    return data - [x_mean_target, y_mean_target, z_mean_target]


def eculidDisSim(x, y):

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
    a_fix = np.load('gsdmd/idlsuf/idesuf_dis350_cbem10.npy')
    target = normzero(a_fix[0:226, 0:3])  # Modify the length according to the actual situation
    # load database
    # source = normzero(np.loadtxt('pc_mov.txt'))  # change to file and use [:,0:3] to extract xyz
    vcb = {0: 0, 1: 11, 2: 14, 3: 16, 4: 18}
    image_target_list = []
    with open('search\\Database\\map_bchhh.txt', 'r') as fp:
        content = fp.readlines()
    str_list = [s.rstrip().split() for s in content]
    image_target_list = [x for x in str_list]
    # score list
    dis_all=[]
    e_all=[]
    m_all=[]
    for i in image_target_list:
        bg=time.time()
        with open(i[0], 'rb') as f:
            datainfo = pickle.load(f)
        source = np.array(datainfo)
        # create a RigidRegistration object
        reg = RigidRegistration(X=target, Y=normzero(source[:, 0:3]))
        # run the registration & collect the results
        TY, (s_reg, R_reg, t_reg) = reg.register()

        #print('TY\n', TY)  # TY is the transformed source points
        #print('s_reg\n', s_reg)  # s_reg the scale of the registration
        #print('R_reg\n', R_reg)  # R_reg the rotation matrix of the registration
        #print('t_reg\n', t_reg)  # t_reg the translation of the registration
        # get the list to coresponding point for charge
        colist = []
        #dislist = []
        TY2=TY
        xi = 0
        for x in target:
            dilist = []
            dilist_r=[]
            for y in TY2:
                dilist.append(eculidDisSim(x, y))
            for y in TY:
                dilist_r.append(eculidDisSim(x, y))
            colist.append([xi, dilist_r.index(min(dilist)), min(dilist)])
            #dislist.append(dilist)
            xi += 1
            TY2=np.delete(TY2, dilist.index(min(dilist)),0)
        # rerange the link between target and source
        #dismtx = list2array(dislist)
        # all distence/ number
        discp = np.array(colist)[:,-1].sum()/len(colist) # distence
        elis = []  # charge
        mlis = []  # Mass
        for ix in colist:
            #discp.append(eculidDisSim(target[ix[0]], TY[ix[1]]))
            elis.append(compE(a_fix[0:226][ix[0]][3], source[ix[1]][3]))
            mlis.append(compM(a_fix[0:226][ix[0]][4], vcb[source[ix[1]][4]]))
        # do score
        dis_all.append(discp)
        e_all.append(elis)
        m_all.append(mlis)
        print(time.time()-bg,i)
    # screen the top candidate
    print(len(dis_all))
    with open('dis_bchhh_cbem350.pkl',"wb") as fs:
        pickle.dump(dis_all, fs)
    print(len(e_all))
    with open('e_bchhh_cbem350.pkl',"wb") as fs:
        pickle.dump(e_all, fs)
    print(len(m_all))
    with open('m_bchhh_cbem350.pkl',"wb") as fs:
        pickle.dump(m_all, fs)

