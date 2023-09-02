import odl
import numpy as np
import pydicom
from multiprocessing.pool import Pool
from skimage.transform import radon,iradon
from skimage.metrics import peak_signal_noise_ratio as psnr,structural_similarity as ssim,mean_squared_error as mse
from cv2 import imwrite


size = 512
n_theta = 120
upsampling_factor = 6
n_s = 512
space = odl.uniform_discr([-128, -128], [128, 128], [size, size], dtype='float32', weighting=1.0)
angle_partition = odl.uniform_partition(0, np.pi, n_theta)
angle_partition_up = odl.uniform_partition(0, np.pi, n_theta*upsampling_factor)
detector_partition = odl.uniform_partition(-120, 120, n_s)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)
geometry_up = odl.tomo.Parallel2dGeometry(angle_partition_up, detector_partition)

RadonSparse = odl.tomo.RayTransform(space, geometry)
FBPSparse = odl.tomo.fbp_op(RadonSparse)

Radon = odl.tomo.RayTransform(space, geometry_up)
FBP = odl.tomo.fbp_op(Radon)
filter_odl = odl.tomo.fbp_filter_op(Radon)
theta = np.linspace(0,360,720)

def write_img(img,name='img1.png'):
    img = (img - img.min())/(img.max() - img.min())
    imwrite(name,255*img)


def reade_ima(path):
    dataset=pydicom.read_file(path)
    img = dataset.pixel_array.astype(np.float32)    ## 读取像素范围 0-4096
    img = img/np.max(img)
    return img

def create_sinogram(img):
    zero = np.zeros([2,512,720])
    ori_sinogram = radon(img,theta)
    ori_sinogram /= ori_sinogram.max()

    filter_sinogram = filter_odl(np.transpose(ori_sinogram,[1,0])).data
    filter_sinogram = np.transpose(filter_sinogram,[1,0])
    filter_sinogram /= filter_sinogram.max()
    zero[0,...] = ori_sinogram
    zero[1,...] = filter_sinogram
    return zero


def bp(sinogram):
    img = iradon(sinogram,theta=theta,filter_name=None)
    return img/img.max()


def filter_op(sinogram):
    filter_sinogram = filter_odl(np.transpose(sinogram,[1,0])).data
    filter_sinogram = np.transpose(filter_sinogram,[1,0])
    filter_sinogram /= filter_sinogram.max()
    return filter_sinogram


def fbp(sinogram):
    filter_sinogram = filter_odl(np.transpose(sinogram,[1,0])).data
    filter_sinogram = np.transpose(filter_sinogram,[1,0])
    filter_sinogram /= filter_sinogram.max()
    img = iradon(filter_sinogram,theta=theta,filter_name=None)
    return img/img.max()

def padding_img(img):
    c,w,h = img.shape
    t = np.max(img.shape)
    pad_size = (t//64 +1)*64
    x_start = (pad_size - w)//2
    y_start = (pad_size - h)//2
    # print("x start:{}, y start:{}".format(x_start,y_start))
    tmp = np.zeros([c,pad_size,pad_size])
    tmp[:,x_start:x_start+w,y_start:y_start+h] = img
    return tmp.astype(np.float32)

def unpadding_img(img):
    c,w,h = img.shape
    x_start = 128
    y_start = 24
    tmp = np.zeros([c,512,720])
    tmp[:,:,:] = img[:,x_start:x_start+512,y_start:y_start+720]
    return tmp.astype(np.float32)


def indicate(img1,img2):
    if len(img1.shape) == 3:
        batch = img1.shape[0]
        psnr0 = np.zeros(batch)
        ssim0 = np.zeros(batch)
        mse0 = np.zeros(batch)
        for i in range(batch):
            t1= img1[i,...]/np.max(img1[i,...])
            t2= img2[i,...]/np.max(img2[i,...])
            psnr0[i,...] = psnr(t1,t2,data_range=1)
            ssim0[i,...] = ssim(t1,t2)
            mse0[i,...] = mse(t1,t2)
        return psnr0,ssim0,mse0
    else:
        img1 /= img1.max()
        img2 /= img2.max()
        psnr0 = psnr(img1,img2,data_range=1)
        ssim0 = ssim(img1,img2)
        mse0 = mse(img1,img2)
        return psnr0,ssim0,mse0

def sinogram_2c_to_img(sinogram2c):
    filter0_sinogram = filter_op(sinogram2c[0,...])
    filter1_sinogram = sinogram2c[1,...]
    average_sinogram = (filter0_sinogram+filter1_sinogram)/2
    sinogram_list = (filter0_sinogram,filter1_sinogram,average_sinogram)
    with Pool(3) as p:
        img_list = p.map(bp,sinogram_list)
    return img_list


# def pathch2img(img,patch_size=320,step=100):
#     batch = ((img.shape[0] - patch_size)//step)
#     tmp = np.zeros([batch**2,patch_size,patch_size])
#     for i in range(batch):
#         for j in range(batch):
            


if __name__ == '__main__':
    sinogram = np.load('/dev/shm/train_sinogram/1.npy')[0,...]
    
    img1 = fbp(sinogram)
    write_img(img1)
    assert 0
    img2 = bp(filter_op(sinogram))
    a = indicate(img1,img2)
    print(a)