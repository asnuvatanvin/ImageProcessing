# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 05:00:46 2022

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

ksize = 5
padding = (ksize-1)//2
img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_REPLICATE)

output_H = (im_H + ksize-1)
output_W = (im_W + ksize-1)

result = np.zeros((output_H,output_W),np.float32)

div = ksize*ksize

for x in range(padding,output_H-padding):
    for y in range(padding,output_W-padding):
        a = 0
        for i in range(-padding,padding+1):
            for j in range(-padding,padding+1):
                a += img[x-i,y-j]
        result[x,y] = a/div
        result[x,y] = result[x,y]/255
        
        
print(result)
plt.imshow(cv2.cvtColor(result,0))
plt.show()
cv2.imwrite('result.png',result)

#plt.imshow(cv2.cvtColor(cv2.blur(img, (ksize,ksize)),0))
#plt.show()