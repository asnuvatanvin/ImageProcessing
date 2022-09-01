# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 02:52:08 2022

@author: u
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def _padding(img,shape,ksize):
    
    im_H = shape[0]
    im_W = shape[1]

    padding = (ksize-1)//2
    padded_img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_REPLICATE)

    output_H = (im_H + ksize-1)
    output_W = (im_W + ksize-1)
    
    return (padded_img,output_H,output_W,padding)

def erosion(img,ksize,structuring_element):
        
    img, output_H, output_W,padding = _padding(img,img.shape,ksize)
    
    result = np.zeros((output_H,output_W),np.uint8)
    
    for x in range(padding,output_H-padding):
        for y in range(padding,output_W-padding):
            temp = img[x-padding:x+padding+1,y-padding:y+padding+1]
            product = temp*structuring_element
            result[x,y] = np.min(product)
            
    return result

def dilation(img,ksize,structuring_element):
    
    img, output_H, output_W, padding = _padding(img,img.shape,ksize)
    
    result = np.zeros((output_H,output_W),np.uint8)
    
    for x in range(padding,output_H-padding):
        for y in range(padding,output_W-padding):
            if(x-padding>=0 and x+padding+1<output_H and y-padding>=0 and y+padding+1<output_W):
                temp = img[x-padding:x+padding+1,y-padding:y+padding+1]
                product = temp*structuring_element
                result[x,y] = np.max(product)
            
    return result

img = cv2.imread('sample3.png',0)

plt.imshow(img,'gray')
plt.show()

_,thres_img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

thres_img //= 255


ksize = 3

structuring_element = np.ones((ksize,ksize),dtype=np.uint8)

#opening

img_erosion = erosion(thres_img,ksize,structuring_element)

img_dilation = dilation(img_erosion,ksize,structuring_element)
 

plt.imshow(img_erosion,'gray')
plt.show()

plt.imshow(img_dilation,'gray')
plt.show()

#closing

img_dilation = dilation(thres_img,ksize,structuring_element)

img_erosion = erosion(img_dilation,ksize,structuring_element)

plt.imshow(img_erosion,'gray')
plt.show()

plt.imshow(img_dilation,'gray')
plt.show()
