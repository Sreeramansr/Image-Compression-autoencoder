# -*- coding: utf-8 -*-
"""
Created on Mon May 18 16:24:42 2020

@author: Sreeraman
"""

# calculation of PSNR values
from math import log10, sqrt 
import cv2 
import numpy as np 
  
def PSNR(original, compressed): 
#    original = original.astype(np.float64) / 255.
#    compressed = compressed.astype(np.float64) / 255.
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 
  
def main(): 
    sum = 0
    for i in range(0,24):
        i = str(i)
        s_zero = i.zfill(1)
        print(s_zero)
        original = cv2.imread("D:/Sree program/Results/No_Dataloss/Decoded_img_7_7_16D_01/image" + s_zero + ".png")
        original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        compressed = cv2.imread("D:/Sree program/Results/No_Dataloss/Decoded_img_7_7_16D_01/reconstruct"+ s_zero + ".png")
        compressed = cv2.cvtColor(compressed, cv2.COLOR_BGR2GRAY)
        value = PSNR(original, compressed) 
        sum = sum + value
        print(f"PSNR value is {value} dB")
    avg = sum/24
    print(avg)
        
       
if __name__ == "__main__": 
    main() 