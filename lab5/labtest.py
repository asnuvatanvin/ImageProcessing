# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 18:07:05 2022

@author: u
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('labtest.png',0)

plt.imshow(img,'gray')
plt.show()

im_H = img.shape[0]
im_W  = img.shape[1]


kernel = np.array(([0,0,0,1,0,0,0],
                   [0,0,1,0,1,0,0],
                   [0,1,0,0,0,1,0],
                   [1,0,0,0,0,0,1],
                   [0,1,0,0,0,1,0],
                   [0,0,1,0,1,0,0],
                   [0,0,0,1,0,0,0]),np.uint8)

ksize = 7
padding = (ksize-1)//2

_,thres_img = cv2.threshold(img, 170, 1, cv2.THRESH_BINARY)

thres_img = cv2.copyMakeBorder(thres_img,padding,padding,padding,padding,cv2.BORDER_CONSTANT,value=0)

output_H = im_H + ksize - 1
output_W = im_W + ksize - 1  

result = np.zeros((output_H,output_W),np.uint8)

for x in range(padding,output_H-padding):
    for y in range(padding,output_W-padding):
        a = -1
        temp = thres_img[x-padding:x+padding+1,y-padding:y+padding+1]
        for i in range(ksize):
            for j in range(ksize):
                if(kernel[i,j]==1 and temp[i,j]!=1):
                    a = 0
                    break
            if(a==0):
                break
        
        if(a==-1):
           result[x,y] = 1
            
plt.imshow(result,'gray')
plt.show()
