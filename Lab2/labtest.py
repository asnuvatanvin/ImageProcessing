# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 01:05:25 2022

@author: u
"""

import cv2
import math
import numpy as np
import matplotlib.pyplot as plt



def gaussian():
    sigma = 5
    
    variance = sigma*sigma
    
    ksize = 7
    padding = (ksize-1)//2
    
    constant = np.sqrt(2*3.1415)*sigma
    
    g = np.zeros((ksize,ksize),np.float32)
    
    for i in range(-padding,padding+1):
        for j in range(-padding,padding+1):
            g[i+padding,j+padding] = np.exp(-(i**2+j**2)/(2*variance))*constant
            
    return g

def gaussian_x():
    sigma = 1.1
    variance = sigma*sigma
    
    ksize = 7
    padding = (ksize-1)//2
    
    div = 2*3.1416*variance*variance
    
    gx = np.zeros((ksize,ksize))
    gy = np.zeros((ksize,ksize))
    
    for i in range(-padding,padding+1):
        for j in range(-padding,padding+1):
            gx[i,j] = -(i/div)*np.exp(-((i**2+j**2)/(2*variance)))
            
    plt.imshow(gx,'gray')
    plt.show()
    return gx

def gaussian_y():
    sigma = 1.1
    variance = sigma*sigma
    
    ksize = 7
    padding = (ksize-1)//2
    
    div = 2*3.1416*variance*variance
    
    g = np.zeros((ksize,ksize))
    
    for i in range(-padding,padding+1):
        for j in range(-padding,padding+1):
            g[i,j] = -(j/div)*np.exp(-((i**2+j**2)/(2*variance)))
    return g;
            
    


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

resultx = np.zeros((output_H,output_W),np.float32)
resulty = np.zeros((output_H,output_W),np.float32)
ans = np.zeros((output_H,output_W))


#gx = np.array(([1,1,1],[0,0,0],[-1,-1,-1]))
#gy = np.array(([-1,0,1],[-1,0,1],[-1,0,1]))

g = gaussian()

'''
plt.imshow(g,'gray')
plt.show()
'''

for x in range(padding,output_H-padding):
    for y in range(padding,output_W-padding):
        a = 0
        b = 0
        c = 0
        ip = img[x,y]
        for i in range(-padding,padding+1):
            for j in range(-padding,padding+1):
                iq = img[x-i,y-j]
                a = g[i+padding,j+padding]*np.exp(-((ip-iq)**2/(2*25)))
                b += a
                c+= a*img[x-i,y-j]
                
        #resultx[x,y] =  a
        #resultx[x,y] /= 255
        #resulty[x,y] =  b
        #resulty[x,y] /= 255
        ans[x,y] = c/b
        
        


'''

for x in range(padding,output_H-padding):
    for y in range(padding,output_W-padding):
        a = 0
        for i in range(-padding,padding+1):
            for j in range(-padding,padding+1):
                a += img[x-i,y-j]
        a /= div
        result[x,y] = a/255
'''

resultx = cv2.normalize(resultx,resultx,0,255,cv2.NORM_MINMAX)
resultx = np.round(resultx).astype(np.uint8)

resulty = cv2.normalize(resulty,resulty,0,255,cv2.NORM_MINMAX)
resulty = np.round(resulty).astype(np.uint8)

ans = cv2.normalize(ans,ans,0,255,cv2.NORM_MINMAX)
ans = np.round(ans).astype(np.uint8)
                
plt.imshow(cv2.cvtColor(resultx,0))
plt.show()

plt.imshow(cv2.cvtColor(resulty,0))
plt.show()


plt.imshow(cv2.cvtColor(ans,0))
plt.show()

                

