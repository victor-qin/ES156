from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import cv2
import copy
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def noise_image(img, noise_scale, N, M):
	# Downsample
	def downsample(im):
		dpix = np.zeros((M/2, N/2))
		for i in range(0, M, 2):
			for j in range(0, N,2):
				dpix[i/2, j/2] = im[i,j]
		return dpix

	# Add noise
	def noise(im):
		# Access pixels
		pix = np.asarray(im, dtype=np.uint8)
		# Add noise
		noisy_img = pix + np.random.normal(0.0, noise_scale, pix.shape)
		# Covert noisy image pixel values to integers
		noisy_img = np.asarray(noisy_img, dtype=np.uint8)
		# Convert noisy image array back to image
		return noisy_img

	return noise(downsample(img))

# Filter a noisy image
def filter_image(img, N, M):

	# Denoise the image with specified method (own created method, numpy or opencv)
	def denoise(im):
		p = np.asarray(im, dtype=np.uint8) + np.zeros((M/2,N/2), dtype=np.uint8)
		for i in range(1, M/2-1):
			for j in range(1, N/2-1):
				neighbors = np.asarray([p[i-1, j], p[i+1, j], p[i, j-1],p[i, j+1], p[i+1, j+1], p[i-1, j+1], p[i+1, j-1], p[i+1, j-1]])
				mindist = min(abs(p[i,j] - n) for n in neighbors)
				std = np.std(neighbors)
				if mindist > std:
					p[i, j] = np.mean(neighbors)

		return p

	# Resize a downsampled image
	def resize(im):
		fimg = np.zeros((M, N))
		for i in range(0, M):
			for j in range(0, N):
				fimg[i, j] = im[i/2,j/2]
		return fimg

	return resize(denoise(img))


# Set smallest fourier coefficients to 0 
def get_alpha_coeffs(coeffs, mags, alpha, N, M):
	ret = copy.copy(coeffs)
	num_removed = int(M*N*(1.0-alpha))
	flatMags = np.ravel(mags)
	smallestMags = np.argsort(flatMags)[:num_removed]
	for i in smallestMags:
		min_idx = np.unravel_index(i, (M,N))
		ret[min_idx] = 0 
	return ret

# Computes mean squared reconstruction error
def compute_MSE(img, img_new):
	error = 0
	for i in range(N):
		for j in range(N):
			error += (img[i][j] - img_new[i][j])**2
	return error/(N**2)
