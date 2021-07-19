# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 19:53:16 2021

@author: ywleo
"""

# ============= Color Space Conversion ==================
# ref: 
# 1. https://blog.csdn.net/qq_30091945/article/details/78236347
# 2. 

import cv2
import numpy as np

def rgb2hsi(rgb_img):
    hg, wd = np.shape(rgb_img)[0], np.shape(rgb_img)[1]
    R,G,B = cv2.split(rgb_img)
    [R,G,B] = [i/255.0 for i in ([R,G,B])]
    
    hsi_img = rgb_img.copy()
    H, S = np.zeros((hg, wd)), np.zeros((hg, wd))
    epsilon = 0.000001
    
    for y in range(hg):
        #Compute H
        denominator = np.sqrt( (R[y]-G[y])**2 + (R[y]-B[y]) * (G[y]-B[y]) )
        numerator = (R[y] - G[y] + R[y] - B[y]) / 2
        theda = np.arccos(numerator / (denominator + epsilon))
        del numerator, denominator
        
        H[y][B[y] <= G[y]] = theda[B[y] <= G[y]] /(2 * np.pi)
        H[y][B[y] > G[y]] = (2 * np.pi - theda[B[y] > G[y]]) /(2 * np.pi)
        
        #Compute S
        mins = []
        for x in range(wd):
            mins.append(np.min([R[y][x], G[y][x], B[y][x]]))
        mins = np.array(mins)
        S[y] = 1 - mins * 3 / (R[y] + G[y] + B[y] + epsilon)
        
    #Compute I
    I = (R + G + B) / 3.0
    
    hsi_img[:,:,0] = H * 255
    hsi_img[:,:,1] = S * 255
    hsi_img[:,:,2] = I * 255
    
    return hsi_img


def hsi2rgb(hsi_img):
    hg, wd = np.shape(hsi_img)[0], np.shape(hsi_img)[1]
    H,S,I = cv2.split(hsi_img)
    [H,S,I] = [i/255.0 for i in ([H,S,I])]
    
    rgb_img = hsi_img.copy()
    R,G,B = H,S,I
    epsilon = 0.000001
    
    for y in range(hg):
        h = H[y] * 2 * np.pi
        #RG sector (0 <= H < 120)
        cond = (h >= 0) & (h < 2 * np.pi / 3)
        denominator = np.cos(np.pi / 3 - h)
        b = I[y] * (1 - S[y])
        r = I[y] * (1 + S[y] * np.cos(h) / (denominator + epsilon))
        g = 3 * I[y] - r - b
        del denominator
        
        R[y][cond], G[y][cond], B[y][cond] = r[cond], g[cond], b[cond]
        
        #GB sector (120 <= H < 240)
        cond = (h >= 2 * np.pi / 3) & (h < 4 * np.pi / 3)
        minus = 2 * np.pi / 3
        denominator = np.cos(np.pi - h)
        r = I[y] * (1 - S[y])
        g = I[y] * (1 + S[y] * np.cos(h - minus) / (denominator + epsilon))
        b = 3 * I[y] - r - g
        del denominator
        
        R[y][cond], G[y][cond], B[y][cond] = r[cond], g[cond], b[cond]
        
        #BR sector (240 <= H < 360)
        cond = (h >= 4 * np.pi / 3) & (h < 2 * np.pi)
        minus = 4 * np.pi / 3
        denominator = np.cos(5 * np.pi / 3 - h)
        g = I[y] * (1 - S[y])
        b = I[y] * (1 + S[y] * np.cos(h - minus) / (denominator + epsilon))
        r = 3 * I[y] - g - b
        del denominator
        
        R[y][cond], G[y][cond], B[y][cond] = r[cond], g[cond], b[cond]
        
    rgb_img[:,:,0] = R * 255
    rgb_img[:,:,1] = G * 255
    rgb_img[:,:,2] = B * 255
    
    return rgb_img.clip(0,255).astype(np.uint8)


def rgb2ycbcr(rgb_img):
    ycrcb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YCR_CB)
    Y, Cr, Cb = cv2.split(ycrcb_img)
    return cv2.merge([Y, Cb, Cr])

def ycbcr2rgb(ycbcr_img):
    Y, Cb, Cr = cv2.split(ycbcr_img)
    ycrcb_img = cv2.merge([Y, Cr, Cb])
    return cv2.cvtColor(ycrcb_img, cv2.COLOR_YCR_CB2RGB)

def rgb2hsv(rgb_img):
    return cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)

def hsv2rgb(hsv_img):
    return cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)

