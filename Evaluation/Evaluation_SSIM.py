# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 07:27:28 2020

@author: Sreeraman
"""
from tensorflow import keras
import tempfile
import tensorflow_model_optimization as tfmot
from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt
import numpy as np
from math import log10, sqrt 
import cv2


def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err
#def compare_images(imageA, imageB, title):
#	# compute the mean squared error and structural similarity
#	# index for the images
#	m = mse(imageA, imageB)
#	s = ssim(imageA, imageB)
#	# setup the figure
#	fig = plt.figure(title)
#	plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
#	# show first image
#	ax = fig.add_subplot(1, 2, 1)
#	plt.imshow(imageA, cmap = plt.cm.gray)
#	plt.axis("off")
#	# show the second image
#	ax = fig.add_subplot(1, 2, 2)
#	plt.imshow(imageB, cmap = plt.cm.gray)
#	plt.axis("off")
#    
#	plt.show()



array =([])
def main():

    
    for i in range(0,24):
        i = str(i)
        s_zero = i.zfill(1)
        print(s_zero)
        original = cv2.imread("D:/Sree program/Results/Dataloss1/Decoded_lossimg_14_14_16D_60%/image" + s_zero + ".png")
        original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        compressed = cv2.imread("D:/Sree program/Results/Dataloss1/Decoded_lossimg_14_14_16D_60%/reconstruct"+ s_zero + ".png")
        compressed = cv2.cvtColor(compressed, cv2.COLOR_BGR2GRAY)
        m = mse(original, compressed)
        s = ssim(original, compressed)
    	# setup the figure
        fig = plt.figure("original vs reconstructed")
        plt.suptitle("SSIM: %.2f" % (s)) #MSE: %.2f, m
        # show first image
        ax = fig.add_subplot(1, 2, 1)
        plt.imshow(original, cmap = plt.cm.gray)
        plt.axis("off")
        # show the second image
        ax = fig.add_subplot(1, 2, 2)
        plt.imshow(compressed, cmap = plt.cm.gray)
        plt.axis("off")
        plt.show()
        fig.savefig("D:/Sree program/Results/Dataloss1/Decoded_lossimg_14_14_16D_60%/SSIM" + s_zero + ".png")

if __name__ == "__main__": 
    main() 
    

       
