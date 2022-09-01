# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 09:59:34 2022

@author: u
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

img = cv2.imread('lena.png',0)

im_H = img.shape[0]
im_W = img.shape[1]

ksize = 9

padding = (ksize-1)//2

img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_REPLICATE)

output_H = (im_H + ksize-1)
output_W = (im_W + ksize-1)

kernel = np.zeros( (ksize,ksize), np.float32)



#kernel = np.zeros((5,5),np.float32)

for i in range(ksize):
  kernel[i,i] = 1

ker = kernel.astype(np.uint8)
plt.imshow(ker, 'gray')

out = signal.convolve2d(img, kernel)

plt.imshow(out,'gray')

plt.title("Output image:")

plt.show()

pad_ud = (out.shape[0]//2 - kernel.shape[0]//2)
pad_lr = (out.shape[1]//2 - kernel.shape[1]//2)

kpad = cv2.copyMakeBorder(kernel, pad_ud, pad_ud-1, pad_lr, pad_lr-1, cv2.BORDER_CONSTANT, 0)

# take image in fourier domain

fimg = np.fft.fft2(out)

fsh = np.fft.fftshift(fimg)

mag = np.abs(fsh)
phase = np.angle(fsh)


# take kernel in fourier domain

himg = np.fft.fft2(kpad)
hsh = np.fft.fftshift(himg)

hmag = np.abs(hsh)

plt.imshow(hmag,'gray')
plt.show()

# deconvolution

omag = np.divide(mag,hmag)

plt.imshow(np.log(omag),'gray')

plt.title("blurred image:")

plt.show()

op = np.multiply(omag , np.exp(1J*phase))

fsh = np.fft.ifftshift(op)
fimg = np.real(np.fft.ifft2(fsh))

plt.imshow(fimg/255,'gray')

plt.title("Output of motion deblur with curtain noise")

plt.show()


