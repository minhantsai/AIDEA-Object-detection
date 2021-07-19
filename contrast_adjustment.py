# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 19:47:33 2021

@author: ywleo
"""

#========== Contrast Adjustment =========================
# ref: 
#https://medium.com/@cindylin_1410#/%E6%B7%BA%E8%AB%87-opencv-
# %E7%9B%B4%E6%96%B9%E5%9C%96%E5%9D%87%E8%A1%A1%E5%8C%96-ahe%E5
#%9D%87%E8%A1%A1-clahe%E5%9D%87%E8%A1%A1-ebc9c14a8f96

import cv2
import numpy as np

def histogram_equalization(img):
    #histogram equalization
    return cv2.equalizeHist(img)

def cla_histogram_equalization(img, tiles=8, ths=40.0):
    # create clahe image
    clahe = cv2.createCLAHE(tileGridSize=(tiles, tiles),clipLimit=ths)
    return clahe.apply(img)

def gamma_transform(img, c=1, gamma=0.4):
    return np.power(c * img, gamma).clip(0,255).astype(np.uint8)
