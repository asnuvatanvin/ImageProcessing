# -*- coding: utf-8 -*-
"""
Created on Wed May 11 11:03:42 2022

@author: u
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('labtest.png',0)

plt.imshow(cv2.cvtColor(img,0))
plt.show()

plt.title(label="Histogram of Input Image",
          fontsize=20,
          color="black")

plt.hist(img.ravel(),256,[0,256])
plt.show()

im_H = img.shape[0]
im_W = img.shape[1]

frame = im_H * im_W

#output = np.zeros_like(img)

output = np.zeros((im_H,im_W))

icount = np.zeros(256)

for i in range(im_H):
    for j in range(im_W):
        intensity = img[i,j]
        icount[intensity] += 1

pdf = icount/frame


cdf = np.zeros(256)
cdf[0] = pdf[0]

for i in range(1,256):
    cdf[i] = cdf[i-1]+pdf[i]

for i in range(im_H):
    for j in range(im_W):
        intensity = img[i,j]
        output[i,j] = np.round(255*cdf[intensity])

        
plt.title(label="CDF of Input Image",
          fontsize=20,
          color="black")
plt.plot(cdf)
plt.show()

output = output.astype(np.uint8)

icounto = np.zeros(256)

for i in range(im_H):
    for j in range(im_W):
        intensity = output[i,j]
        icounto[intensity] += 1

pdfo = icounto/frame

cdfo = np.zeros(256)
cdfo[0] = pdfo[0]

for i in range(1,256):
    cdfo[i] = cdfo[i-1]+pdfo[i]
    
plt.imshow(cv2.cvtColor(output,0))
plt.show()
    
plt.title(label="Histogram of Output Image",
          fontsize=20,
          color="black")
plt.hist(output.ravel(),256,[0,256])
plt.show()
    
plt.title(label="CDF of Output Image",
          fontsize=20,
          color="black")
plt.plot(cdfo)
plt.show()




