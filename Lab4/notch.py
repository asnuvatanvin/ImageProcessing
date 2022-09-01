# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 08:51:03 2022

@author: u
"""

import cv2 
import matplotlib.pyplot as plt
import numpy as np
'''

xx = []
yy = []

def click_event(event,x,y,a,b):
    if event == cv.EVENT_LBUTTONDOWN:
        xx.append(y)
        yy.append(x)
        

path = 'pi.jpg'

img = cv.imread(path)

img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

plt.imshow(img,'gray')

plt.show()

d0=25
n=1

imx = np.fft.fft2(img)

sft = np.fft.fftshift(imx)

phase = np.angle(sft)

magx = np.abs(sft)

k = 0

while 1:
    if k==3:
        break
    magx = np.log(magx)
    
    k+=1
    
cv.imshow('image',magx)

cv.setMouseCallback('image', click_event)

cv.waitKey(0)

cv.destroyAllWindows()

cv.imshow('image',magx)

cv.setMouseCallback('image', click_event)

cv.waitKey(0)

cv.destroyAllWindows()

cv.imshow('image',magx)

cv.setMouseCallback('image', click_event)

cv.waitKey(0)

cv.destroyAllWindows()

cv.imshow('image',magx)

cv.setMouseCallback('image', click_event)

cv.waitKey(0)

cv.destroyAllWindows()

m = img.shape[0]
n = img.shape[1]

notch = np.zeros((m,n),np.float32)

for u in range(m):
    for v in range(n):
        prod = 1
        for k in range(4):
            d = np.sqrt((u-m//2-(xx[k]-m//2))**2+(v-n//2-(yy[k]-n//2))**2)
            dk = np.sqrt((u-m//2+(xx[k]-m//2))**2+(v-n//2+(yy[k]-n//2))**2)
            prod*=1/ ( ( 1+(d0/d)**(2*n)  )*( 1+(d0/dk)**(2*n) ) )
        notch[u][v] = prod
        
plt.imshow(notch,'gray')

plt.show()

mag = np.abs(sft)

mag = mag*notch

op = np.multiply(mag,np.exp(1j*phase))

op  =np.fft.ifftshift(op)

op = np.real(np.fft.ifft2(op))

plt.imshow(op,'gray')

plt.show()
'''

xx = []
yy= []

img = cv2.imread('pi.jpg',0)

plt.imshow(img,'gray')
plt.show()

imx = np.fft.fft2(img)

shift = np.fft.fftshift(imx)

phase = np.angle(shift)

mag = np.abs(shift)

k = 0

def click_event(event,x,y,a,b):
    if event == cv2.EVENT_LBUTTONDOWN:
        xx.append(y)
        yy.append(x)

while 1:
    if k==3:
        break;
    mag = np.log(mag)
    
    k += 1
    
cv2.imshow('image',mag)

cv2.setMouseCallback('image', click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('image',mag)

cv2.setMouseCallback('image', click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('image',mag)

cv2.setMouseCallback('image', click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('image',mag)

cv2.setMouseCallback('image', click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()

d0 = 25
n = 1

im_H = img.shape[0]
im_W = img.shape[1]

centerx = im_H//2
centery = im_W//2

notch = np.zeros((im_H,im_W))

for i in range(im_H):
    for j in range(im_W):
        prod = 1
        for k in range(4):
            u = i - centerx
            uk = xx[k] - centerx
            v = j - centery
            vk = yy[k] - centery
            
            d = np.sqrt((u-uk)**2+(v-vk)**2)
            dk = np.sqrt((u + uk)**2+(v+vk)**2)
            
            prod *= (1 / (1+(d0/d)**(2*n)))*(1 / (1 + (d0/dk)**(2*n)))
            
        notch[u,v] = prod
        
new = np.abs(shift)*notch

newim = np.multiply(new,np.exp(1j*phase))

ans = np.real(np.fft.ifft2(np.fft.ifftshift(newim)))

plt.imshow(ans,'gray')
plt.show()


