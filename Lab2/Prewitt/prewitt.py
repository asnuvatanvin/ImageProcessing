# -*- coding: utf-8 -*-
"""
Created on Tue May 10 21:57:02 2022

@author: u
"""

import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

#take input image
img = cv2.imread("rubiks_cube.png",cv2.IMREAD_GRAYSCALE)
plt.imshow(cv2.cvtColor(img,0))
plt.show()

im_H = img.shape[0]
im_W = img.shape[1]


ksize = 3
padding = (ksize-1)//2
img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_REPLICATE)

output_H = (im_H + ksize-1)
output_W = (im_W + ksize-1)

gx = np.zeros((output_H,output_W),np.float32)
gy = np.zeros((output_H,output_W),np.float32)
g = np.zeros((output_H,output_W),np.float32)


hx = np.array(([-1,0,1],
              [-1,0,1],
              [-1,0,1]),np.float32)

hy = np.array(([-1,-1,-1],
              [0,0,0],
              [1,1,1]),np.float32)
 
for x in range(padding,output_H-padding):
    for y in range(padding,output_W-padding):
        a = 0
        b = 0
        for i in range(-padding,padding+1):
            for j in range(-padding,padding+1):
                a += hx[i+padding,j+padding]*img[x-i,y-j]
                b += hy[i+padding,j+padding]*img[x-i,y-j]
        gx[x,y] = a
        gy[x,y] = b
        g[x,y] = np.sqrt(a**2+b**2)
       
        
cv2.normalize(gx, gx, 0, 255, cv2.NORM_MINMAX)
cv2.normalize(gy, gy, 0, 255, cv2.NORM_MINMAX)
cv2.normalize(g, g, 0, 255, cv2.NORM_MINMAX)
        

gx = np.round(gx).astype(np.uint8)
gy = np.round(gy).astype(np.uint8)
g = np.round(g).astype(np.uint8)



plt.imshow(cv2.cvtColor(gx,0))
plt.show()

plt.imshow(cv2.cvtColor(gy,0))
plt.show()

plt.imshow(cv2.cvtColor(g,0))
plt.show()


cv2.imwrite('result.png',g)




