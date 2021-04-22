import numpy as np
# sample depth value from dataset
# 
# YUV little-endian order reference:
# https://docs.microsoft.com/en-us/windows/win32/medfound/10-bit-and-16-bit-yuv-video-formats

def ERPSample(num):
    # sample depth dataset: ClassroomVideo
    dt = np.dtype('<u2')
    YUV = np.fromfile('ERP_sampleDepth.yuv', dtype = dt)
    Y_Component = YUV[0:4096*2048]
    return Y_Component[num]

def PerspectiveSample(num):
    #  sample depth dataset: IntelFrog 
    dt = np.dtype('<u2')
    YUV = np.fromfile('Perspective_sampleDepth.yuv', dtype = dt)
    Y_Component = YUV[0:1920*1080]
    return Y_Component[num]