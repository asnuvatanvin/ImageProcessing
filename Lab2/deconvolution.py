#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 00:22:35 2022

@author: asnuvatanvin
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pypher

def normalize(img):
    nImg = np.zeros(img.shape)#, dtype='uint8')
    
    max_ = img.max()
    min_ = img.min()
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            nImg[i][j] = (img[i][j]-min_)/(max_-min_) * 255
    
    return np.array(nImg, dtype='uint8')

def zero_pad(image, shape, position='corner'):
   
    difh = shape[0] - psf.shape[0]
    difw = shape[1] - psf.shape[1]
    
    pad_H = difh // 2
    pad_W = difw // 2
    
    if difh % 2 != 0:
        pad_img = cv2.copyMakeBorder(psf, pad_H + 1, pad_H, pad_W + 1, pad_W, cv2.BORDER_CONSTANT,value = 0)
    else:
        pad_img = cv2.copyMakeBorder(psf, pad_H, pad_H, pad_W, pad_W, cv2.BORDER_CONSTANT,value = 0)
    
    return pad_img


def psf2otf(psf, shape):

    if np.all(psf == 0):
        return np.zeros_like(psf)

    inshape = psf.shape
    # Pad the PSF to outsize
    psf = zero_pad(psf, shape, position='corner')
    
    f = np.fft.fft2(psf)
    otf = np.fft.fftshift(f)
    
    plt.imshow(np.abs(np.flip(otf,axis=1)),'gray')
    plt.show()

    return np.abs(np.flip(otf,axis=1))

def butterworth(d0,n,shape):
    
    im_H = shape[0]
    im_W = shape[1]
    
    centerx = im_H//2
    centery = im_W//2
    
    
    g = np.zeros((shape),np.float32)
    
    for i in range(im_H):
        for j in range(im_W):
            u = i - centerx
            v = j - centery
            p = 1+(((u**2+v**2)**0.5)/d0)**(2*n)
            q = 1/p
            g[i,j] = q
    return g

   
def motion_blurr(img):
    
    im_H = img.shape[0]
    im_W = img.shape[1]

    ksize = 3
    padding = (ksize-1)//2
    img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_REPLICATE)

    output_H = (im_H + ksize-1)
    output_W = (im_W + ksize-1)

    result = np.zeros((output_H,output_W),np.float32)
    
    #ksize = 15
    
    motion_blurr_filter = np.zeros((ksize,ksize),np.uint8)
    
    for i in range(ksize):
        for j in range(ksize):
            if(i==j):
                motion_blurr_filter[i,j]=1
        
    plt.imshow(motion_blurr_filter,'gray')
    plt.show()
    
    result = cv2.filter2D(img,ddepth=-1,kernel=motion_blurr_filter)
    
    return (result,motion_blurr_filter)


img = cv2.imread('lena.png',0)

cv2.imwrite("Input.png",img)


plt.imshow(img,'gray')
plt.show()

blurred_image,psf = motion_blurr(img)

plt.imshow(blurred_image,'gray')
plt.show()
cv2.imwrite("Blurred.png",blurred_image)

f = np.fft.fft2(blurred_image)
fshift = np.fft.fftshift(f)

magnitude = np.abs(fshift)
phase = np.angle(fshift)

shape = blurred_image.shape

otf = psf2otf(psf,blurred_image.shape)

newmagnitude = magnitude/otf

d0 = 15
n = 10

bfilter = butterworth(d0, n, shape)

newmagnitude *= bfilter

newimg = np.multiply(newmagnitude,np.exp(1j*phase))

spatialoutput = np.real(np.fft.ifft2(np.fft.ifftshift(newimg)))

plt.imshow(normalize(spatialoutput),'gray')
plt.show()

cv2.imwrite("Output.png",normalize(spatialoutput))


