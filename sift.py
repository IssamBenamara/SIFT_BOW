import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tools import *

def compute_grad(I):
    h_x = 0.5*np.asarray([1,0,-1])
    h_y = 0.5*np.asarray([-1,-2,-1])
    Iy = conv_separable(I, h_x, h_y)
    Ix = conv_separable(I, h_y, h_x)
    return Ix, Iy

def compute_grad_mod_ori(I):
    Ix, Iy = compute_grad(I)
    Gm = np.sqrt(Ix**2+Iy**2)
    Go = compute_grad_ori(Ix, Iy, Gm)
    return Gm, Go

def compute_sift_region(Gm, Go, mask=None):
    # Note: to apply the mask only when given, do:
    R = np.zeros((16,8))
    Gn = Gm.copy()
    if mask is not None:
        Gn = np.multiply(Gn, mask)
    for i in range(4):
        for j in range(4):
            Gn_region = Gn[j*4:j*4+4,i*4:i*4+4]
            Go_region = Go[j*4:j*4+4,i*4:i*4+4]

            for v in range(8):
                ind = np.where(Go_region==v)
                s = np.sum(Gn_region[ind[0],ind[1]])
                R[i*4+j][v] = s
    sift = R.ravel()
    norm = np.linalg.norm(sift)
    if norm < 0.5:
        return np.zeros(sift.shape)
    sift = sift/np.linalg.norm(sift)
    sift = np.clip(sift,0,0.2)
    sift = sift/np.linalg.norm(sift)
    return sift

def compute_sift_image(I):
    x, y = dense_sampling(I)
    im = auto_padding(I)
    
    sifts = np.zeros((len(x), len(y), 128))
    Gn, Go = compute_grad_mod_ori(im)
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            Gn_region = Gn[xi:xi+16,yj:yj+16]
            Go_region = Go[xi:xi+16,yj:yj+16]
            sifts[i, j, :] = compute_sift_region(Gn_region, Go_region) # TODO SIFT du patch de coordonnee (xi, yj)
    return sifts