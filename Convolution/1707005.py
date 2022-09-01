# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 21:51:45 2022

@author: u
"""

import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from scipy import signal


#take input image
img = cv2.imread("rsz_lena.png",cv2.IMREAD_GRAYSCALE)


im_H = img.shape[0]
im_W = img.shape[1]


#filter for convolution
kernel = np.array(([0,-1,0],[-1,5,-1],[0,-1,0]),np.float32)

ksize_H = kernel.shape[0]
ksize_W = kernel.shape[1]

padding_H = (ksize_H-1)//2
padding_W = (ksize_W-1)//2

output_H = im_H + ksize_H - 1
output_W = im_W + ksize_W - 1
output_shape = (output_H,output_W)

#Add padding to image
img = cv2.copyMakeBorder(img, padding_H, padding_H, padding_W, padding_W, cv2.BORDER_REPLICATE)

#Add padding to image
nfilter = cv2.copyMakeBorder(kernel,output_H-ksize_H,0,0,output_W-ksize_W,cv2.BORDER_CONSTANT,value=0)


#create toeplitz matrices
toeplitz_list = []

for i in range(nfilter.shape[0]-1,-1,-1):
    column = nfilter[i,:]
    row = np.r_[column[0],np.zeros(output_W-1)]
    toeplitz_matrix = toeplitz(column,row)
    toeplitz_list.append(toeplitz_matrix)


#create indices for doubly blocked matrix
column = range(1, nfilter.shape[0]+1)
row = np.r_[column[0], np.zeros(output_H-1,dtype=int)]
doubly_indices = toeplitz(column,row)



#create doubly blocked matrix
h = toeplitz_list[0].shape[0]*doubly_indices.shape[0]
w = toeplitz_list[0].shape[1]*doubly_indices.shape[1]
doubly_blocked = np.zeros((h,w))

toeplitz_h, toeplitz_w = toeplitz_list[0].shape


for i in range(doubly_indices.shape[0]):
    for j in range(doubly_indices.shape[1]):
        start_i = i * toeplitz_h
        start_j = j * toeplitz_w
        end_i = start_i + toeplitz_h
        end_j = start_j + toeplitz_w
        doubly_blocked[start_i:end_i,start_j:end_j] = toeplitz_list[doubly_indices[i,j]-1]


#convert image matrix to image vector
def matrix_to_vector(image,output_shape):
    output_vector = np.zeros(output_shape[0]*output_shape[1])
    image = np.flipud(image)
    for i in range(output_shape[0]):
        start = i * output_shape[1]
        end = start + output_shape[1]
        row = image[i,:]
        output_vector[start:end] = row
    return output_vector


vectorized_image = matrix_to_vector(img, output_shape)
 

#matrix multiplication       
result_vector = np.matmul(doubly_blocked,vectorized_image)


#convert result vector to matrix
def vector_to_image(vector,output_shape):
    output_image = np.zeros((output_shape[0],output_shape[1]),dtype = np.float32)
    for i in range(output_shape[0]):
        start = i * output_shape[1]
        end = start + output_shape[1]
        output_image[i,:] = vector[start:end]
    return output_image



output = vector_to_image(result_vector, output_shape)
output = np.flipud(output)
print(output)


plt.imshow(cv2.cvtColor(output,0))
plt.show()


result = signal.convolve2d(img,kernel,'full')
print(result)


plt.imshow(cv2.cvtColor(result,0))
plt.show()



        
        

