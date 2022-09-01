# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 01:41:23 2022

@author: u
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('sample1.png',0)

plt.imshow(img,'gray')
plt.show()

_,thres_img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

thres_img //= 255

im_H = img.shape[0]
im_W = img.shape[1]

print(thres_img)

ksize = 15
padding = (ksize-1)//2
img = cv2.copyMakeBorder(thres_img, padding, padding, padding, padding, cv2.BORDER_REPLICATE)

output_H = (im_H + ksize-1)
output_W = (im_W + ksize-1)

result = np.zeros((output_H,output_W),np.float32)

structuring_element = np.ones((ksize,ksize),dtype=np.uint8)
 
for x in range(padding,output_H-padding):
    for y in range(padding,output_W-padding):
        temp = img[x-padding:x+padding+1,y-padding:y+padding+1]
        product = temp*structuring_element
        result[x,y] = np.min(product)
      

plt.imshow(result,'gray')
plt.show()
