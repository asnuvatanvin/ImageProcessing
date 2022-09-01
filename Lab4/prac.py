# -*- coding: utf-8 -*-
"""
Created on Wed May 25 09:57:07 2022

@author: u
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def clip(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] < 0:
                img[i][j] = 0
            if img[i][j] > 255:
                img[i][j] = 255
    return img.astype(np.float32)

def normalize(img):
    nImg = np.zeros(img.shape)#, dtype='uint8')
    
    max_ = img.max()
    min_ = img.min()
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            nImg[i][j] = (img[i][j]-min_)/(max_-min_) * 255
    
    
    return np.array(nImg, dtype='uint8')

def gaussianhomomorphic(gl, gh, c, d0, shape):
    
    im_H = shape[0]
    im_W = shape[1]
    centerx = im_H//2
    centery = im_W//2
    g = np.zeros((shape),np.float32)
    
    ds = d0**2
    
    for i in range(im_H):
        for j in range(im_W):
            u = i - centerx
            v = j - centery
            a = 1 - np.exp(((-c*(u**2+v**2))/ds))
            g[i,j] = (gh - gl) * a + gl
    return g
            
            

img = cv2.imread('wiki.jpg',0)

cv2.imshow('Input',img)

logimage = np.log1p(img)
f = np.fft.fft2(logimage)

cv2.imwrite('cmag.png',normalize(np.abs(f)))
cv2.imwrite('phase.png',normalize(np.angle(f)))

fshift = np.fft.fftshift(f)

magnitude = np.abs(fshift)
phase = np.angle(fshift)

gl = 0.5
gh = 1.2
c = 0.1
#d0 = 2*np.pi*sigma**2
d0 = 50
#sigma = 10.0

shape = img.shape

ffilter = gaussianhomomorphic(gl, gh, c, d0, shape)

plt.imshow(ffilter,'gray')
plt.show()
cv2.imwrite('filter.png',normalize(ffilter))

newmagnitude = np.multiply(magnitude,ffilter)

newimg = np.multiply(newmagnitude,np.exp(1j*phase))

spatialoutput = np.real(np.fft.ifft2(np.fft.ifftshift(newimg)))

output = np.expm1(spatialoutput)

print(output)

plt.imshow(output,'gray')

cv2.imshow('Output',normalize(output))
cv2.imwrite('Output.png',normalize(output))

cv2.waitKey(0)
cv2.destroyAllWindows()


