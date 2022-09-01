# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 01:47:27 2022

@author: u
"""

import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

#take input image
img = cv2.imread("rubiks_cube.png",cv2.IMREAD_GRAYSCALE)
plt.imshow(cv2.cvtColor(img,0))
plt.show()

im_H = img.shape[0]
im_W = img.shape[1]


def gaussian(sigma,img,ksize,padding):
    gfilter = np.zeros((ksize,ksize),np.float32)
    div = (sigma*sigma)*2
    for i in range(-padding,padding+1):
        for j in range(-padding,padding+1):
            gfilter[i+padding,j+padding] = math.exp(-((i**2+j**2)/div))
    return gfilter



ksize = 5
padding = (ksize-1)//2
img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_REPLICATE)

output_H = (im_H + ksize-1)
output_W = (im_W + ksize-1)

result = np.zeros((output_H,output_W),np.float32)

sigma = 5

div = (sigma*sigma)*2

gaussian_filter = gaussian(sigma,img,ksize,padding)

for x in range(padding,output_H-padding):
    for y in range(padding,output_W-padding):
        a = 0
        fil = 0
        normalization = 0
        ip = img[x,y]
        for i in range(-padding,padding+1):
            for j in range(-padding,padding+1):
                iq = img[x-i,y-j]
                fil = gaussian_filter[i+padding,j+padding]*(math.exp(-(((ip-iq)**2)/div)))
                normalization += fil
                a += fil*img[x-i,y-j]
        result[x,y] = a/normalization
        result[x,y] /= 255
        


plt.imshow(cv2.cvtColor(result,0))
plt.show()