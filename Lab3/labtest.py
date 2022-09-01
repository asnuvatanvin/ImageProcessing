# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 20:52:30 2022

@author: u
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def match(s,cdff):
    for j in range(256):
        if(s==cdff[j]):
            return j
        elif(s<cdff[j]):
            a = np.abs(s - cdff[j-1])
            b = np.abs(s - cdff[j])
            if(a<b):
                return j-1
            else:
                return j
    return 255
        
def erlang():
    k = 9
    miu = 5
    
    kk = 1
    
    for i in range(1,k+1):
        kk *= i
        
    g = np.empty(shape=256)
        
    for i in range(256):
        g[i] = ((i**k-1) * np.exp(-(i/miu)))/((miu**k)*kk)
        
    plt.plot(g)
    plt.show()
        
    return g
        
        

def gaussian():
    miu = 30
    sigma = 80
    
    g = np.empty(shape=256)
    
    variance = sigma*sigma
    constant = 1 / np.sqrt(2*3.1416) * sigma
    
    for i in range(256):
        g[i] = np.exp(-(((i-miu)**2)/(2*variance))) * constant
        
    return g

    


img = cv2.imread('labtest.png',0)

plt.imshow(img,'gray')
plt.show()

im_H = img.shape[0]
im_W = img.shape[1]

frame = im_H * im_W

plt.hist(img.ravel(),256,[0,256])
plt.show()


icount = np.zeros((256))

for i in range(im_H):
    for j in range(im_W):
        icount[img[i,j]] += 1
        
pdf = icount/frame

cdf = np.zeros((256),np.float32)

cdf[0] = pdf[0]

for i in range(1,256):
    cdf[i] = cdf[i-1] + pdf[i]
    
cdf *= 255
    
plt.plot(cdf)
plt.show()

result = np.zeros((im_H,im_W))

'''

for i in range(im_H):
    for j in range(im_W):
        result[i,j] = np.round(cdf[img[i,j]]*255)
'''
g = erlang()
pdff = g/np.sum(g)


'''
for i in range(256):
    summ += icountt[i]

pdff = icount/summ
'''

cdff = np.zeros((256))

cdff[0] = pdff[0]

for i in range(1,256):
    cdff[i] = cdff[i-1] + pdff[i]
    
plt.plot(cdff)
plt.show()

cdff *= 255

cdff = np.round(cdff).astype(np.uint8)

for i in range(im_H):
    for j in range(im_W):
        intensity = img[i,j]
        ans = match(np.round(cdf[intensity]),cdff)
        result[i,j] = ans
        


plt.imshow(result,'gray')
plt.show()

plt.hist(result.ravel(),256,[0,256])
plt.show()

