# -*- coding: utf-8 -*-
"""
Created on Mon May 23 23:10:29 2022

@author: u
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def search(a,arr):
    for i in range(256):
        if(a == arr[i]):
            return i
        elif (a < arr[i]):
            b = arr[i]
            c = arr[i-1]
            if((b-a)>(a-c)):
                return i-1
            else:
                return i
    return 255

def gaussian(miu,sigma):
    variance = sigma*sigma
    constant = 1/(np.sqrt(2*3.1416)*sigma)
    g = np.empty(shape=256)
    for i in range (256):
        g[i] = np.exp(-((i-miu)**2)/(2*variance))*constant
    return g
        
    

u1,sigma1 = [int(x) for x in input('Enter the values of miu1 and sigma1:').split()]
u2,sigma2 = [int(x) for x in input('Enter the values of miu2 and sigma2:').split()]

img = cv2.imread("histogram.jpg",cv2.IMREAD_GRAYSCALE)

plt.imshow(img,cmap='gray')
plt.show()

#cv2.imshow('Input',img)

plt.title(label="Histogram of Input Image",
          fontsize=20,
          color="black")
plt.hist(img.ravel(),256,[0,256])
plt.show()


im_H = img.shape[0]
im_W = img.shape[1]

frame = im_H*im_W


intensities = np.zeros(256)
cdf = np.zeros(256,np.float32)
equalized_input = np.zeros((im_H,im_W),np.uint8)
output = np.zeros((im_H,im_W),np.uint8)

for i in range(im_H):
    for j in range(im_W):
        intensities[img[i,j]] += 1
        
pdf = intensities/frame
cdf[0] = pdf[0]


for i in range(1,len(pdf)):
    cdf[i] = cdf[i-1] + pdf[i]
    
cdf *= 255

plt.title(label="CDF of Input Image",
          fontsize=20,
          color="black")
plt.plot(cdf)
plt.show()

'''
for i in range(im_H):
    for j in range(im_W):
        equalized_input[i,j] = np.round(cdf[img[i,j]])

        
cv2.imwrite('Equalized.png',equalized_input)
'''

g1 = gaussian(u1, sigma1)
g2 = gaussian(u2,sigma2)



plt.title(label="Gaussian Function 1",
          fontsize=20,
          color="black")
plt.plot(g1)
plt.show()

plt.title(label="Gaussian Function 2",
          fontsize=20,
          color="black")
plt.plot(g2)
plt.show()

g = g1 + g2
plt.title(label="Bimodal Function",
          fontsize=20,
          color="black")
plt.plot(g)
plt.show()

frameg = np.sum(g)

pdfg = g/frameg

cdfg = np.zeros(256)

cdfg[0] = pdfg[0]

for i in range(1,len(pdf)):
    cdfg[i] = cdfg[i-1] + pdfg[i]
    
cdfg *= 255

plt.title(label="CDF of Bimodal Function",
          fontsize=20,
          color="black")
plt.plot(cdfg)
plt.show()


cdfg = np.round(cdfg).astype(np.uint8)

for i in range(im_H):
    for j in range(im_W):
        a = np.round(cdf[img[i,j]])
        b = search(a,cdfg)
        output[i,j] = b

#cv2.imwrite('Output.png',output)

#cv2.waitKey(0)
#cv2.destroyAllWindows() 

#print(output)

plt.imshow(output,cmap='gray')
plt.show()

plt.title(label="Histogram of Output Image",
          fontsize=20,
          color="black")
plt.hist(output.ravel(),256,[0,256])
plt.show()

ointensities = np.zeros(256)

for i in range(im_H):
    for j in range(im_W):
        ointensities[output[i,j]] += 1
        
pdf = ointensities/frame
cdf[0] = pdf[0]


for i in range(1,len(pdf)):
    cdf[i] = cdf[i-1] + pdf[i]
    
cdf *= 255

plt.title(label="CDF of Output Image",
          fontsize=20,
          color="black")
plt.plot(cdf)
plt.show()

