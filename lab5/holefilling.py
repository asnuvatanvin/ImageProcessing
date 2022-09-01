# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 16:29:27 2022

@author: u
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('sample4.png',0)

im_H = img.shape[0]
im_W = img.shape[1]

_,thres_img = cv2.threshold(img, 100, 1, cv2.THRESH_BINARY)

ksize = 5
padding = (ksize - 1)//2

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (ksize,ksize))


thres_img = cv2.copyMakeBorder(thres_img, padding, padding, padding, padding, cv2.BORDER_CONSTANT,value=0)


plt.imshow(thres_img,'gray')
plt.show()

p = 0
q = 0

for i in range(im_H):
    for j in range(im_W):
        if(thres_img[i,j]==1 and i+1<im_H and thres_img[i+1,j]==0):
            p = i+1
            q = j
            break
    if(p!=0 or q!=0):
        break
            
print(p,q)
print(thres_img[p,q])

output_H = im_H + ksize - 1
output_W = im_W + ksize - 1

result = np.zeros((output_H,output_W),np.uint8)
x = np.zeros((output_H,output_W),np.uint8)
x[p,q] = 1

ca = 1 - thres_img
y = x


for d in range(4):
    
    op = cv2.dilate(y,kernel)
    
    for i in range(output_H):
        for j in range(output_W):
            if(ca[i,j]==1 and op[i,j]==1):
                x[i,j]=1
    '''        
    if(np.sum(y)==np.sum(x)):
        print(d)
        break
    y = x
    '''

for i in range(output_H):
    for j in range(output_W):
        if(x[i,j]==1 or thres_img[i,j]==1):
            result[i,j]=1
            
plt.imshow(result,'gray')
plt.show()