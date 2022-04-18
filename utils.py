import cv2
import numpy as np
from copy import copy

def getImageFeatures(img, colorSpace, types=1):
    
    # Getting features depending on colorSpace
    if colorSpace == 'RGB':
        b = img[:,:,0].ravel()
        g = img[:,:,1].ravel()
        r = img[:,:,2].ravel()
        out = np.stack((b,g,r))
        
    elif colorSpace == 'Grayscale':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        out = np.expand_dims(img.ravel(),axis=0)
        
    elif colorSpace == 'LAB':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L = img[:,:,0].ravel()
        A = img[:,:,1].ravel()
        B = img[:,:,2].ravel()
        out = np.stack((L,A,B))
    
    if types==1:   # Not adding x,y values
        out = out
    elif types==2: # Adding x,y values
        x = np.argwhere(np.ones((img.shape[0],img.shape[1])))[:,1]
        y = np.argwhere(np.ones((img.shape[0],img.shape[1])))[:,0]
        out = np.append(out, [x,y], axis=0)
    
    # Normalizing
    out = out.astype(np.float32)
    std = np.std(out,axis=1)
    for i in range(out.shape[0]):
        out[i] = out[i]/std[i]
        
    return out.T

def colorSegmentedImage(img,seg):
    out_im = copy(img)
    for class_ in range(int(np.max(seg)+1)):
        pix_indexes = np.argwhere(seg==class_)
        out_im[pix_indexes[:,0],pix_indexes[:,1]] = np.mean(img[pix_indexes[:,0],pix_indexes[:,1]],axis=0)
    return out_im