# -*- coding: utf-8 -*-
"""
Created on Sun May  1 00:26:56 2022

@author: u
"""


import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

#take input image
img = cv2.imread("moon.png",cv2.IMREAD_GRAYSCALE)
plt.imshow(cv2.cvtColor(img,0))
plt.show()

im_H = img.shape[0]
im_W = img.shape[1]


ksize = 3
padding = (ksize-1)//2
img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_REPLICATE)

output_H = (im_H + ksize-1)
output_W = (im_W + ksize-1)

result = np.zeros((output_H,output_W),np.float32)

laplacian = np.array(([-1,-1,-1],[-1,8,-1],[-1,-1,-1]),np.float32)

for x in range(padding,output_H-padding):
    for y in range(padding,output_W-padding):
        a = 0
        for i in range(-padding,padding+1):
            for j in range(-padding,padding+1):
                a += laplacian[i+padding,j+padding]*img[x-i,y-j]
        result[x,y] = a/255
        

result = img + result

cv2.normalize(result, result, 0, 255, cv2.NORM_MINMAX)
        
result = np.round(result).astype(np.uint8)

print(result)
plt.imshow(cv2.cvtColor(result,0))
plt.show()

cv2.imwrite('result.png',result)


