import numpy as np
import cv2 
from matplotlib import pyplot as plt

def gaussian_homomorphic(gh,gl,c,d0,shape):
    
    im_H = shape[0]
    im_W = shape[1]
    
    centerx = im_H//2
    centery = im_W//2
    
    ds = d0**2
    
    g = np.zeros((shape[0],shape[1],2),np.float32)
    
    for i in range(im_H):
        for j in range(im_W):
            u = i - centerx
            v = j - centery
            r = -c*((u**2+v**2)/(ds))
            k = 1 - np.exp(r)
            g[i,j] = (gh - gl) * k + gl
    return g


def normalize(img):
    min_ = img.min()
    max_ = img.max()
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i,j] = ((img[i,j]-min_)/(max_-min_))*255
            
    img = img.astype(np.uint8)
    
    return img



img = cv2.imread('wiki.jpg',0)

plt.imshow(img,'gray')
plt.show()

logimg = np.log1p(img)

dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
cv2.imwrite("Magnitude.png",normalize(np.log(cv2.magnitude(dft[:,:,0],dft[:,:,1]))))

dft_shift = np.fft.fftshift(dft)



magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
plt.imshow(magnitude_spectrum,cmap='gray')
plt.title("magnitude")
plt.show()


phase = cv2.phase(dft_shift[:,:,0],dft_shift[:,:,1])
plt.imshow(phase,cmap='gray')
plt.title("phase")
plt.show()

shape = img.shape


gh = 1.2
gl = 0.5
c = 0.1
d0 = 50

ffilter = gaussian_homomorphic(gh, gl, c, d0, shape)

conv = dft_shift*ffilter

#ans = cv2.merge([conv,phase])

f_ishift = np.fft.ifftshift(conv)

img_back = cv2.idft(f_ishift)

img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

output = normalize(img_back)

plt.imshow(output,'gray')
plt.show()
cv2.imwrite("Output.png",img_back)
