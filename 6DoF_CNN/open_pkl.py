# -*- coding: UTF8 -*-

# cPickle是python2系列用的，3系列已經不用了，直接用pickle就好了
import numpy as np
import matplotlib
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt
import struct
import pandas
import os
import glob
import pickle
# import cv2
# from skimage.measure import block_reduce
from skimage.transform import resize


def PerspectiveSample(yuv_path=''):
    dt = np.dtype('<u2')
    YUV = np.fromfile(yuv_path, dtype = dt)
    Y_sample= YUV[0]
    return Y_sample
def perspective_depth_convert(depth, num_bits, zNear, zFar):
    a = zFar / ( zFar - zNear )
    b = ( zFar * zNear ) / ( zNear - zFar )
    Z = ( pow(2,num_bits) * b ) / ( depth - pow(2,num_bits) * a)
    return Z

path = "/home/tsehou/6DoF_CNN/code_silver/raw_datasets/source_views/perspective/yuv/"
print('IntelFrog:')
print(PerspectiveSample(f'{path}/IntelFrog/depth/1/50.yuv'))
print(perspective_depth_convert(PerspectiveSample(f'{path}/IntelFrog/depth/1/50.yuv'),16,0.3,1.62))
print('OrangeKitchen:')
print(PerspectiveSample(f'{path}/OrangeKitchen/depth/00/16.yuv'))
print(perspective_depth_convert(PerspectiveSample(f'{path}/OrangeKitchen/depth/00/16.yuv'),16,2.2,7.2))
print('PoznanCarpark:')
print(PerspectiveSample(f'{path}/PoznanCarpark/depth/0/40.yuv'))
print(perspective_depth_convert(PerspectiveSample(f'{path}/PoznanCarpark/depth/0/40.yuv'),16,34.5064,2760.511))
print('PoznanFencing')
print(PerspectiveSample(f'{path}/PoznanFencing/depth/00/40.yuv'))
print(perspective_depth_convert(PerspectiveSample(f'{path}/PoznanFencing/depth/00/40.yuv'),16,3.5,7.0))
print('PoznanHall')
print(PerspectiveSample(f'{path}/PoznanHall/depth/0/80.yuv'))
print(perspective_depth_convert(PerspectiveSample(f'{path}/PoznanHall/depth/0/80.yuv'),16,18.5064,2760.511))
print('PoznanStreet')
print(PerspectiveSample(f'{path}/PoznanStreet/depth/0/40.yuv'))
print(perspective_depth_convert(PerspectiveSample(f'{path}/PoznanStreet/depth/0/40.yuv'),16,34.5064,2760.511))
print('TechnicolorPainter')
print(PerspectiveSample(f'{path}/TechnicolorPainter/depth/0/50.yuv'))
print(perspective_depth_convert(PerspectiveSample(f'{path}/TechnicolorPainter/depth/0/50.yuv'),16, 1.808811,5.402476))



# 重點是rb和r的區別，rb是開啟2進位制檔案，文字檔案用r
# dict_keys(['fn_frames', 'imgs', 'depth', 'c_para', 'fn_sv'])
# f=open('/home/tsehou/6DoF_CNN/code_silver/prepare_datasets/pickle/PoznanHall.pkl','rb')
# data=pickle.load(f)
# print(data.keys())
# print(data['depth'][0][0][0][0][0])
# print(get_depth(18.5064,2760.511,10))

