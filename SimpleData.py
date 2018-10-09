# -*- coding: utf-8 -*-

import os
from PIL import Image
import numpy as np

#读取文件夹mnist下的42000张图片，图片为灰度图，所以为1通道，
#如果是将彩色图作为输入,则将1替换为3，并且data[i,:,:,:] = arr改为data[i,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
def load_data():
    data = np.empty((41900,1,28,28),dtype="float32")
    label = np.empty((41900,),dtype="uint8")

    imgs = os.listdir("./trial")
    num = len(imgs)
    for i in range(num):
        img = Image.open("./trial/"+imgs[i])
        arr = np.asarray(img,dtype="float32")
        data[i,:,:,:] = arr
        label[i] = int(imgs[i].split('.')[0])
    data = data.reshape(41900,28,28,1)
    return data,label

def load_data_test():
    data = np.empty((100,1,28,28),dtype="float32")
    label = np.empty((100,),dtype="uint8")

    imgs = os.listdir("./test")
    num = len(imgs)
    for i in range(num):
        img = Image.open("./test/"+imgs[i])
        arr = np.asarray(img,dtype="float32")
        data[i,:,:,:] = arr
        label[i] = int(imgs[i].split('.')[0])
    data = data.reshape(100,28,28,1)
    return data,label