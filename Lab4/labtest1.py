# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 18:51:53 2022

@author: u
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('filterf.jpg',0)

plt.imshow(img,'gray')
plt.show()

im_H = img.shape[0]
im_W  = img.shape[1]

def normalize(img):
    
    im_H = img.shape[0]
    im_W = img.shape[1]
    
    mx = img.max()
    mn = img.min()
    
    for i in range(im_H):
        for j in range(im_W):
            img[i,j] = ((img[i,j]-mn)/(mx-mn))*255.0
            
    return (np.array(img,dtype='uint8'))

def gaussian_high_pass(im_H,im_W):
    
    gh = 1.5
    gl = 0.5
    c = 0.5
    d0 = 64
    
    g = np.zeros((im_H,im_W))
    centerx = im_H//2
    centery = im_W//2
    ds = d0**2
    
   
    
    for i in range(im_H):
        for j in range(im_W):
            u = i-centerx
            v = j-centery
            a = 1 - np.exp((-c*(u**2+v**2))/ds)
            g[i,j] = (gh-gl)*a+gl
            
    plt.imshow(g,'gray')
    plt.show()
    return g
def gaussian_low_pass(im_H,im_W):
    gh = 1.5
    gl = 0.5
    c = 0.5
    d0 = 64
    
    g = np.zeros((im_H,im_W))
    centerx = im_H//2
    centery = im_W//2
    ds = d0**2
    

    for i in range(im_H):
        for j in range(im_W):
            u = i-centerx
            v = j-centery
            a = np.exp((-c*(u**2+v**2))/ds)
            g[i,j] = (gh-gl)*a+gl
            
    plt.imshow(g,'gray')
    plt.show()
    return g

def butterworth_low_pass(im_H,im_W):
    gh = 1.5
    gl = 0.5
    d0 = 64
    b = np.zeros((im_H,im_W))
    centerx = im_H//2
    centery = im_W//2
    n=8
    nn = 2*n
    
    for i in range(im_H):
        for j in range(im_W):
            u = i-centerx
            v = j-centery
            a = 1/(1+(np.sqrt(u**2+v**2)/d0)**nn)
            b[i,j] = (gh-gl)*a+gl
            
    plt.imshow(b,'gray')
    plt.show()
    return b

def butterworth_high_pass(im_H,im_W):
    gh = 1.5
    gl = 0.5
    d0 = 64
    n=8
    nn = 2*n
    
    b = np.zeros((im_H,im_W),np.float32)
    centerx = im_H//2
    centery = im_W//2
    
    for i in range(im_H):
        for j in range(im_W):
            u = i-centerx
            v = j-centery
            if(u == 0):
                u = 0.00001
            if(v==0):
                v = 0.00001
            a = 1/(1+(d0/np.sqrt(u**2+v**2))**nn)
            b[i,j] = (gh-gl) * a + gl
            
    plt.imshow(b,'gray')
    plt.show()
    return b
    
a = butterworth_high_pass(im_H,im_W)
b = butterworth_low_pass(im_H,im_W)
c = gaussian_high_pass(im_H,im_W)
d = gaussian_low_pass(im_H, im_W)

logimage = np.log1p(img)

dftshift = np.fft.fft2(logimage)
centershift = np.fft.fftshift(dftshift)

mag = np.abs(centershift)
phase = np.angle(centershift)

plt.imshow(mag,'gray')
plt.show()
plt.imshow(phase,'gray')
plt.show()

gl = mag*d
gh = mag*c
bl = mag*b
bh = mag*a

newimage = np.multiply(bh,np.exp(1j*phase))

inversedimage = np.real(np.fft.ifft2(np.fft.ifftshift(newimage)))

ans = np.expm1(inversedimage)

result = img + ans

plt.imshow(result,'gray')
plt.show()

