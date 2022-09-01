import cv2
import numpy as np
import matplotlib.pyplot as plt

def normalize(img):
    nImg = np.zeros(img.shape)#, dtype='uint8')
    
    max_ = img.max()
    min_ = img.min()
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            nImg[i][j] = (img[i][j]-min_)/(max_-min_) * 255
    
    #print(img.min(), img.max())
    #print(nImg.min(), nImg.max())
    
    return np.array(nImg, dtype='uint8')

img =cv2.imread('wiki.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('Input Image', img)

logimage = np.log1p(img)
f = np.fft.fft2(logimage)
fshift = np.fft.fftshift(f)


magnitude = normalize(np.log(np.abs(fshift)))
cv2.imshow('Magnitude Spectrum', magnitude)

# Gaussian filter
height, width = img.shape
#height, width = (5,5)
sig = 20

gauss = np.zeros((height, width))

#def __gaussian_filter(self, I_shape, filter_params):#crea un highpass filter
P = height/2
Q = width/2
U, V = np.meshgrid(range(height), range(width), sparse=False, indexing='ij')
Duv = (((U-P)**2+(V-Q)**2)**(1/2)).astype(np.dtype('d'))
c=1
D0=10
h = np.exp((-c*(Duv**2)/(2*(D0**2))))#lowpass filter
H=(1-h)


cv2.imshow('Gaussian filter', H)

# Filtering
mag = np.abs(fshift)
angle = np.angle(fshift)

new_mag = mag * H
#cv2.imshow('New Magnitude', normalize(new_mag))

combined = np.multiply(new_mag, np.exp(1j*angle))
imgCombined = np.real(np.fft.ifft2(np.fft.ifftshift(combined)))

output = np.exp(imgCombined) 

plt.imshow(output,'gray')
plt.show()

cv2.imshow('Output Image', normalize(output))

cv2.waitKey(0)
cv2.destroyAllWindows()