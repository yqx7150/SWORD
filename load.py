# for i in range(1, 4800):
#     a = np.load('C:/Users/huang/Desktop/train_img/',i)
#     # # mat
#     # savemat('a.mat', mdict={'data': a})
#     c = loadmat('a.mat')["data"]
#     # npy
#     np.save(f"b.npy", a)
    # d = np.load('a.npy')

    # assert np.allclose(c, d)
    # img
    # plt.imshow(d[0,:,:],cmap=plt.get_cmap('gray'))
    # plt.savefig('img.png')
    # plt.show()
import numpy as np
import scipy.io as io
import os
# def npy_mat(npy_path,mat_path):
#     npyname_path = os.listdir(npy_path)
#     for npyname in npyname_path:
#         npyname = os.path.join(npy_path,npyname)
#         name = npyname[:-4]
#         name = name[39:]
#         mat_name = name+'.mat'
#         mat_name = os.path.join(mat_path,mat_name)
#         npy = np.load(npyname)
#         io.savemat(mat_name,{'data':npy})
# npy_mat(r'/home/lqg/hb/CT_CODE/DOSE_HANKEL/train_img2/',r'/home/lqg/hb/CT_CODE/DOSE_HANKEL/train_au/')


import numpy as np
import scipy.io as sio
data=sio.loadmat('/home/lqg/hb/CT_CODE/DOSE_HANKEL/data_aug/1.mat')
np.save('/home/lqg/hb/CT_CODE/DOSE_HANKEL/data_aug2/1.npy',data)

# def mat_npy(mat_path,npy_path):
#     matname_path = os.listdir(mat_path)
#     for matname in matname_path:
#         matname = os.path.join(mat_path,matname)
#         name = matname[:-4]
#         name = name[39:]
#         npy_name = name+'.npy'
#         npy_name = os.path.join(npy_path,npy_name)
#         mat = sio.loadmat(matname)
#         np.save(npy_name,{'data':mat})
# mat_npy(r'/home/lqg/hb/CT_CODE/DOSE_HANKEL/data_aug/',r'/home/lqg/hb/CT_CODE/DOSE_HANKEL/data_augnp/')