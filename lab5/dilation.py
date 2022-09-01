# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 01:54:48 2022

@author: u
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('sample2.png',0)

plt.imshow(img,'gray')
plt.show()

_,thres_img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

thres_img //= 255

im_H = img.shape[0]
im_W = img.shape[1]


ksize = 3
padding = (ksize-1)//2
img = cv2.copyMakeBorder(thres_img, padding, padding, padding, padding, cv2.BORDER_REPLICATE)

output_H = (im_H + ksize-1)
output_W = (im_W + ksize-1)

result = np.zeros((output_H,output_W),np.uint8)

structuring_element = np.array([[0,1,0],[1,1,1],[0,1,0]])
 

for x in range(padding,output_H-padding):
    for y in range(padding,output_W-padding):
        
        temp = img[x-padding:x+padding+1,y-padding:y+padding+1]
        product = temp*structuring_element
        result[x,y] = np.max(product)

plt.imshow(result,'gray')
plt.show()

