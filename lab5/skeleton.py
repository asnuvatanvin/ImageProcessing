# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 13:57:08 2022

@author: u
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
def dilation(img,se):
    
    ksize = 3
    padding = 1
    
    result = np.zeros((output_H,output_W),np.uint8)
    
    for x in range(padding,output_H-padding):
        for y in range(padding,output_W-padding):
            temp = thres_img[x-padding:x+padding+1,y-padding:y+padding+1]
            prod = temp*structuring_element
            result[x,y] = prod.max()
            
    return result;

'''

def erosion(img,se):
    
    ksize = se.shape[1]
    padding = (ksize - 1)//2
    output_H = img.shape[0] + ksize -1
    output_W = img.shape[1] + ksize - 1
    
    result = np.zeros((output_H,output_W),np.uint8)
    
    print(ksize)
    print(padding)

    for x in range(padding,output_H-padding):
        for y in range(padding,output_W-padding):
            temp = thres_img[x-padding:x+padding+1,y-padding:y+padding+1]
            a = 0
            for i in range(ksize):
                for j in range(ksize):
                    if(se[i,j]==1 and temp[i,j]!=1):
                        a=2
                        break
                if(a==2):
                    break
            if(a==0):
                result[x,y]=1
            a=0
    return result


img = cv2.imread('skeleton.bmp',0)

im_H = img.shape[0]
im_W = img.shape[1]


plt.imshow(img, 'gray')
plt.show()

thres_img = np.ones((im_H,im_W))


for i in range(im_H):
    for j in range(im_W):
        if(img[i,j]>=170):
            thres_img[i,j] = 0


print(thres_img)
            
         
#_,thres_img = cv2.threshold(img,150,1,cv2.THRESH_BINARY)



plt.imshow(thres_img, 'gray')
plt.show()

cv2.imwrite("input.png",thres_img)

'''
kernel = np.array(([0,0,0,1,0,0,0],
                   [0,0,1,0,1,0,0],
                   [0,1,0,0,0,1,0],
                   [1,0,0,0,0,0,1],
                   [0,1,0,0,0,1,0],
                   [0,0,1,0,1,0,0],
                   [0,0,0,1,0,0,0]),np.uint8)
'''

kernel = np.array(([1,1,1],[1,1,1],[1,1,1]),np.uint8)

ksize = kernel.shape[0]
padding = 1

thres_img = cv2.copyMakeBorder(thres_img,padding,padding,padding,padding,cv2.BORDER_CONSTANT,value=0)


output_H = im_H + ksize - 1
output_W = im_W + ksize - 1

#result  = dilation(thres_img,structuring_element)
        
#plt.imshow(result,'gray')
#plt.show()

#lib = cv2.dilate(thres_img,structuring_element)
#lib = cv2.erode(thres_img,kernel)
#lib = cv2.morphologyEx(thres_img,cv2.MORPH_CLOSE,kernel)
#lib = cv2.morphologyEx(thres_img,cv2.MORPH_OPEN,kernel)

#plt.imshow(lib,'gray')
#plt.show()



skeleton = np.zeros((output_H,output_W),np.uint8)
AerodeB = thres_img

for p in range(10):
    
    op = cv2.morphologyEx(AerodeB, cv2.MORPH_OPEN, kernel)
    
    diff = AerodeB - op
    
    
    for i in range(output_H):
        for j in range(output_W):
            if(diff[i,j]==1):
                skeleton[i,j]=1
                
    AerodeB = cv2.erode(AerodeB,kernel)
    
    if(AerodeB.max()==0):
        break

plt.imshow(skeleton, 'gray')
plt.show()

cv2.imwrite("Skeleton.png",skeleton)


