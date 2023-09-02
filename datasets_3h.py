# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Return training and evaluation/test datasets from config fiTles."""
import random

import jax
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import cv2
from torch import normal
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.io import loadmat
import numpy as np
import os
import odl
from scipy import misc
#from scipy.misc.pilutil import imrotate
import torch
from scipy.io import loadmat, savemat
from DWT_IDWT.DWT_IDWT_layer import DWT_1D, DWT_2D, IDWT_1D, IDWT_2D

Fan_angle_partition = odl.uniform_partition(0, 2 * np.pi, 720)
Fan_detector_partition = odl.uniform_partition(-360, 360, 720)
Fan_geometry = odl.tomo.FanBeamGeometry(Fan_angle_partition, Fan_detector_partition,
                            src_radius=500, det_radius=500)
Fan_reco_space = odl.uniform_discr(min_pt=[-128, -128], max_pt=[128, 128], shape=[512, 512], dtype='float32')
Fan_ray_trafo = odl.tomo.RayTransform(Fan_reco_space, Fan_geometry)
Fan_filter = odl.tomo.fbp_filter_op(Fan_ray_trafo)


class GetCT(Dataset):

    def __init__(self, root, augment=None):
        super().__init__()
        self.data_names = np.array([root+"/"+x  for x in os.listdir(root)])
        self.augment = None

    def img_normalized(self,img):
        return (img - np.min(img))/(np.max(img) - np.min(img))

    def rescale(self,img):
        normal_img = (img - np.min(img))/(np.max(img) - np.min(img))
        return (normal_img *2. -1)

    def padding_img(self,img):
        w,h = img.shape
        h1 = (h//64 + 1)*64
        tmp = np.zeros([h1,h1], dtype=np.float32)
        x_start = int((h1 -w)//2)
        y_start = int((h1 -h)//2)
        tmp[x_start:x_start+w,y_start:y_start+h] = img
        return tmp


    def __getitem__(self,index):

        def dwt_data(data):
            ###论文给了两种代码实现，这里用的第一种
            use_pytorch_wavelet = 0
            if not use_pytorch_wavelet:
                # 2位小波变换 一种比较简单的小波 haar(哈尔小波）
                dwt = DWT_2D("haar")
                iwt = IDWT_2D("haar")
            else:
                dwt = DWTForward(J=1, mode='zero', wave='haar').cuda()
                iwt = DWTInverse(mode='zero', wave='haar').cuda()
            ##小波变换
            xll, xlh, xhl, xhh = dwt(data)
            ####论文输出小波变换为以下值
            dwt_data = torch.cat([xll, xlh, xhl, xhh], dim=1)  # [1, 4, 256, 256]
            dwt_data = np.squeeze(dwt_data)
            #####以下内容为保存小波变换的结果为mat文件
            # filepath = "./temp_data/"
            ####tensor转numpy
            # xll = xll.cpu().detach().numpy()
            # xlh = xlh.cpu().detach().numpy()
            # xhl = xhl.cpu().detach().numpy()
            # xhh = xhh.cpu().detach().numpy()
            # dwt_data = dwt_data.cpu().detach().numpy()
            # savemat("xll.mat", {'xll':xll})
            # savemat("xlh.mat", {'xlh':xlh})
            # savemat("xhl.mat", {'xhl':xhl})
            # savemat("xhh.mat", {'xhh':xhh})
            # savemat("dwt_data.mat", {'dwt_data': dwt_data})
            # return [xll,xlh,xhl,xhh]
            return dwt_data


        # img = loadmat(self.data_names[index])["data"]
        img = np.load(self.data_names[index])
        sinogram = Fan_ray_trafo(img).data
        # sinogram = sinogram / sinogram.max()
        #########################aug##########################
        sinogram = np.squeeze(sinogram)

        ########################################################
        sinogram = self.padding_img(sinogram)
        #ori_data = torch.tensor(sinogram).cuda()

        ori_data = sinogram[None, None, ...]
        #print(ori_data.shape)
        # ori_data = np.float32(ori_data)
        ori_data = torch.from_numpy(ori_data).cuda()  # this is good , maybe torch version is different

        
        sinogram = dwt_data(ori_data)

        return sinogram[1:, ...] # 3h


    def __len__(self):
        return len(self.data_names)

def get_data_scaler(config):
  """Data normalizer. Assume data are always in [0, 1]."""
  if config.data.centered:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x


def get_data_inverse_scaler(config):
  """Inverse data normalizer."""
  if config.data.centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x


# def crop_resize(image, resolution):
#   """Crop and resize an image to the given resolution."""
#   crop = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
#   h, w = tf.shape(image)[0], tf.shape(image)[1]
#   image = image[(h - crop) // 2:(h + crop) // 2,
#           (w - crop) // 2:(w + crop) // 2]
#   image = tf.image.resize(
#     image,
#     size=(resolution, resolution),
#     antialias=True,
#     method=tf.image.ResizeMethod.BICUBIC)
#   return tf.cast(image, tf.uint8)


def resize_small(image, resolution):
  """Shrink an image to the given resolution."""
  h, w = image.shape[0], image.shape[1]
  ratio = resolution / min(h, w)
  h = tf.round(h * ratio, tf.int32)
  w = tf.round(w * ratio, tf.int32)
  return tf.image.resize(image, [h, w], antialias=True)


def central_crop(image, size):
  """Crop the center of an image to the given size."""
  top = (image.shape[0] - size) // 2
  left = (image.shape[1] - size) // 2
  return tf.image.crop_to_bounding_box(image, top, left, size, size)


def get_dataset(config, uniform_dequantization=False, evaluation=False):
  """Create data loaders for training and evaluation.

  Args:
    config: A ml_collection.ConfigDict parsed from config files.
    uniform_dequantization: If `True`, add uniform dequantization to images.
    evaluation: If `True`, fix number of epochs to 1.

  Returns:
    train_ds, eval_ds, dataset_builder.
  """
  # Compute batch size for this worker.
  batch_size = config.training.batch_size if not evaluation else config.eval.batch_size
  if batch_size % jax.device_count() != 0:
    raise ValueError(f'Batch sizes ({batch_size} must be divided by'
                     f'the number of devices ({jax.device_count()})')

  # Reduce this when image resolution is too large and data pointer is stored
  shuffle_buffer_size = 10000
  prefetch_size = tf.data.experimental.AUTOTUNE
  num_epochs = None if not evaluation else 1

  # Create dataset builders for each dataset.
  if config.data.dataset == 'CIFAR10':
    dataset_builder = tfds.builder('cifar10')
    train_split_name = 'train'
    eval_split_name = 'test'

    def resize_op(img):
      img = tf.image.convert_image_dtype(img, tf.float32)
      return tf.image.resize(img, [config.data.image_size, config.data.image_size], antialias=True)

  elif config.data.dataset == 'SVHN':
    dataset_builder = tfds.builder('svhn_cropped')
    train_split_name = 'train'
    eval_split_name = 'test'

    def resize_op(img):
      img = tf.image.convert_image_dtype(img, tf.float32)
      return tf.image.resize(img, [config.data.image_size, config.data.image_size], antialias=True)

  elif config.data.dataset == 'CELEBA':
    #dataset_builder = tfds.builder('celeb_a')
    train_split_name = 'train'
    eval_split_name = 'validation'

    def resize_op(img):
      img = tf.image.convert_image_dtype(img, tf.float32)
      img = central_crop(img, 140)
      img = resize_small(img, config.data.image_size)
      return img

  #elif config.data.dataset == 'LSUN':
  #  dataset_builder = tfds.builder(f'lsun/{config.data.category}')
  #  train_split_name = 'train'
  #  eval_split_name = 'validation'

  #  if config.data.image_size == 128:
  #    def resize_op(img):
  #      img = tf.image.convert_image_dtype(img, tf.float32)
  #      img = resize_small(img, config.data.image_size)
  #      img = central_crop(img, config.data.image_size)
  #      return img

  #  else:
  #    def resize_op(img):
  #      img = crop_resize(img, config.data.image_size)
  #      img = tf.image.convert_image_dtype(img, tf.float32)
  #      return img
        
  elif config.data.dataset == 'LSUN':
    #dataset_builder = tfds.builder(f'lsun/{config.data.category}')
    train_split_name = 'train'
    eval_split_name = 'validation'

    def resize_op(img):
      img = crop_resize(img, config.data.image_size)
      img = tf.image.convert_image_dtype(img, tf.float32)
      return img

  elif config.data.dataset in ['FFHQ', 'CelebAHQ']:
    dataset_builder = tf.data.TFRecordDataset(config.data.tfrecords_path)
    train_split_name = eval_split_name = 'train'

  else:
    raise NotImplementedError(
      f'Dataset {config.data.dataset} not yet supported.')

  # Customize preprocess functions for each dataset.
  if config.data.dataset in ['FFHQ', 'CelebAHQ']:
    def preprocess_fn(d):
      sample = tf.io.parse_single_example(d, features={
        'shape': tf.io.FixedLenFeature([3], tf.int64),
        'data': tf.io.FixedLenFeature([], tf.string)})
      data = tf.io.decode_raw(sample['data'], tf.uint8)
      data = tf.reshape(data, sample['shape'])
      data = tf.transpose(data, (1, 2, 0))
      img = tf.image.convert_image_dtype(data, tf.float32)
      if config.data.random_flip and not evaluation:
        img = tf.image.random_flip_left_right(img)
      if uniform_dequantization:
        img = (tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.) / 256.
      return dict(image=img, label=None)

  else:
    def preprocess_fn(d):
      """Basic preprocessing function scales data to [0, 1) and randomly flips."""
      img = resize_op(d['image'])
      if config.data.random_flip and not evaluation:
        img = tf.image.random_flip_left_right(img)
      if uniform_dequantization:
        img = (tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.) / 256.

      return dict(image=img, label=d.get('label', None))


  def create_dataset(dataset_builder, split):
    dataset_options = tf.data.Options()
    dataset_options.experimental_optimization.map_parallelization = True
    dataset_options.experimental_threading.private_threadpool_size = 48
    dataset_options.experimental_threading.max_intra_op_parallelism = 1
    read_config = tfds.ReadConfig(options=dataset_options)
    if isinstance(dataset_builder, tfds.core.DatasetBuilder):
      dataset_builder.download_and_prepare()
      ds = dataset_builder.as_dataset(
        split=split, shuffle_files=True, read_config=read_config)
    else:
      ds = dataset_builder.with_options(dataset_options)
    ds = ds.repeat(count=num_epochs)
    ds = ds.shuffle(shuffle_buffer_size)
    ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds.prefetch(prefetch_size)

  #train_ds = create_dataset(dataset_builder, train_split_name)
  #eval_ds = create_dataset(dataset_builder, eval_split_name)
  
  # dataset = GetCT(root= "./data_augnp/ug")
  # test_dataset = GetCT(root= "./train_data")
  dataset = GetCT(root= "./train_img")
  test_dataset = GetCT(root= "./train_img")
  
  
  train_ds = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True,
                                num_workers=0)
                                
  eval_ds = DataLoader(test_dataset, batch_size=config.eval.batch_size, shuffle=True,
                                 num_workers=0, drop_last=True)
  
  
  
  
  return train_ds, eval_ds #, dataset_builder
