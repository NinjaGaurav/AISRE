import cv2
from PIL import Image 
import numpy as np
import numba as nb
import os
import pickle
import sys
import concurrent.futures
from agec import *
from gaussian2d import gaussian2d
from gettestargs import gettestargs
from hashkey import hashkey
from math import floor, ceil
from scipy import interpolate
from multiprocessing import Process, freeze_support
import warnings
import time

warnings.filterwarnings("ignore")

R = 2
patchsize = 11
gradientsize = 9
Qangle = 24
Qstrength = 3
Qcoherence = 3
trainpath = 'test'
resultpath= 'results'
# Calculate the margin
maxblocksize = max(patchsize, gradientsize)
margin = floor(maxblocksize/2)
patchmargin = floor(patchsize/2)
gradientmargin = floor(gradientsize/2)
# Matrix preprocessing
# Preprocessing normalized Gaussian matrix W for hashkey calculation
weighting = gaussian2d([gradientsize, gradientsize], 2)
weighting = np.diag(weighting.ravel())

with open('filter.p', "rb") as fp:
	h = pickle.load(fp)

def join(image):
	name = image.split('.')[0]
	name = "results"+ name[4:]
	buf = 8
	im1 = cv2.imread(name + '$1_result.bmp')
	h, w, _ = im1.shape
	im1 = im1[0:h-buf, 0:w]

	im2 = cv2.imread(name + '$2_result.bmp')
	h, w, _ = im2.shape
	im2 = im2[buf:h-buf, 0:w]

	im3 = cv2.imread(name + '$3_result.bmp')
	h, w, _ = im3.shape
	im3 = im3[buf:h-buf, 0:w]

	im4 = cv2.imread(name + '$4_result.bmp')
	h, w, _ = im4.shape
	im4 = im4[buf:h-buf, 0:w]

	im5 = cv2.imread(name + '$5_result.bmp')
	h, w, _ = im5.shape
	im5 = im5[buf:h-buf, 0:w]

	im6 = cv2.imread(name + '$6_result.bmp')
	h, w, _ = im6.shape
	im6 = im6[buf:h-buf, 0:w]

	im7 = cv2.imread(name + '$7_result.bmp')
	h, w, _ = im7.shape
	im7 = im7[buf:h-buf, 0:w]

	im8 = cv2.imread(name + '$8_result.bmp')
	h, w, _ = im8.shape
	im8 = im8[buf:h, 0:w]

	im_v = cv2.vconcat([im1, im2, im3, im4, im5, im6, im7, im8])
	cv2.imwrite( name + "_result.bmp", im_v)

def split(image):
    img = Image.open(image)
    width, height = img.size
    upper = 0
    left = 0
    slice_size = int(ceil(height/8))
    slices = int(ceil(height/slice_size))
    buf = 8
    count = 1
    lower = 0
    for slice in range(slices):
        #if we are at the end, set the lower bound to be the bottom of the image
        if count == slices:
            lower = height
        else:
            lower = count*slice_size + buf

        bbox = (left, upper, width, lower)
        working_slice = img.crop(bbox)
        upper += slice_size
        #save the slice
        working_slice.save( os.path.join('results/', os.path.splitext(os.path.basename(image))[0] + "$" + str(count)+".bmp") )
        count +=1  

def upscale(image):
	global R
	global patchsize
	global gradientsize
	global Qangle
	global Qstrength
	global Qcoherence
	global trainpath
	global resultpath
	global maxblocksize
	global margin
	global patchmargin
	global gradientmargin
	global weighting
	global h
	origin = cv2.imread(image)
	# Extract only the luminance in YCbCr
	ycrcvorigin = cv2.cvtColor(origin, cv2.COLOR_BGR2YCrCb)
	grayorigin = ycrcvorigin[:,:,0]
	# Normalized to [0,1]
	grayorigin = cv2.normalize(grayorigin.astype('float'), None, grayorigin.min()/255, grayorigin.max()/255, cv2.NORM_MINMAX)
	# Upscale (bilinear interpolation)
	heightLR, widthLR = grayorigin.shape
	heightgridLR = np.linspace(0,heightLR-1,heightLR)
	widthgridLR = np.linspace(0,widthLR-1,widthLR)
	bilinearinterp = interpolate.interp2d(widthgridLR, heightgridLR, grayorigin, kind='linear')
	heightgridHR = np.linspace(0,heightLR-0.5,heightLR*2)
	widthgridHR = np.linspace(0,widthLR-0.5,widthLR*2)
	upscaledLR = bilinearinterp(widthgridHR, heightgridHR)
	# Calculate predictHR pixels
	heightHR, widthHR = upscaledLR.shape
	predictHR = np.zeros((heightHR-2*margin, widthHR-2*margin))
	operationcount = 0
	totaloperations = (heightHR-2*margin) * (widthHR-2*margin)
	for row in range(margin, heightHR-margin):
		for col in range(margin, widthHR-margin):
			# Get patch
			patch = upscaledLR[row-patchmargin:row+patchmargin+1, col-patchmargin:col+patchmargin+1]
			patch = patch.ravel()
			# Get gradient block
			gradientblock = upscaledLR[row-gradientmargin:row+gradientmargin+1, col-gradientmargin:col+gradientmargin+1]
			# Calculate hashkey
			gy, gx = np.gradient(gradientblock)
			angle, strength, coherence = hashkey(Qangle, weighting, gy, gx)
			# Get pixel type
			pixeltype = ((row-margin) % R) * R + ((col-margin) % R)
			predictHR[row-margin,col-margin] = patch.dot(h[angle,strength,coherence,pixeltype])
	# Scale back to [0,255]
	predictHR = np.clip(predictHR.astype('float') * 255., 0., 255.)
	# Bilinear interpolation on CbCr field
	result = np.zeros((heightHR, widthHR, 3))
	y = ycrcvorigin[:,:,0]
	bilinearinterp = interpolate.interp2d(widthgridLR, heightgridLR, y, kind='linear')
	result[:,:,0] = bilinearinterp(widthgridHR, heightgridHR)
	cr = ycrcvorigin[:,:,1]
	bilinearinterp = interpolate.interp2d(widthgridLR, heightgridLR, cr, kind='linear')
	result[:,:,1] = bilinearinterp(widthgridHR, heightgridHR)
	cv = ycrcvorigin[:,:,2]
	bilinearinterp = interpolate.interp2d(widthgridLR, heightgridLR, cv, kind='linear')
	result[:,:,2] = bilinearinterp(widthgridHR, heightgridHR)
	result[margin:heightHR-margin,margin:widthHR-margin,0] = predictHR
	result = cv2.cvtColor(np.uint8(result), cv2.COLOR_YCrCb2RGB)
	cv2.imwrite('results/' + os.path.splitext(os.path.basename(image))[0] + '_result.bmp', (cv2.cvtColor(result, cv2.COLOR_RGB2BGR)) )

def main():

	mainlist = []
	for parent, dirnames, filenames in os.walk(trainpath):
		for filename in filenames:
			if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
				mainlist.append(os.path.join(parent, filename))

	with concurrent.futures.ProcessPoolExecutor() as executor:
		hexcnt=1
		for mainimg in mainlist:
			# print(mainimg)	#test/1.png  test/apple.jpeg
			start_time = time.time()
			split(mainimg)	#in results apple$1_result.bmp 1$1_result.bmp
			prefix = mainimg.split('.')[0]
			# print(prefix)	test/apple
			prefix = prefix[5:]
			# print(prefix) apple
			imagelist = []
			for parent, dirnames, filenames in os.walk(resultpath):
				for filename in filenames:
					if filename.endswith((prefix+'$1.bmp',prefix+'$2.bmp',prefix+'$3.bmp',prefix+'$4.bmp',prefix+'$5.bmp',prefix+'$6.bmp',prefix+'$7.bmp',prefix+'$8.bmp')):
						imagelist.append(os.path.join(parent, filename))

			print('Processing image#'+str(hexcnt) +', please wait ☻')
			pro=12.5
			for _ in executor.map(upscale,imagelist):
				print("\n"+ str(pro) + " percent done.")
				pro+=12.5

			join(mainimg)
			# apple_result
			nightimg = cv2.imread('results/'+prefix+'_result.bmp')
			cv2.imwrite('results/'+prefix+'_result.bmp',agec_init(nightimg))

			for items in imagelist:
				os.remove(items)
				os.remove(items.split('.')[0]+"_result.bmp")
			print("--- %s seconds ---" % (time.time() - start_time))
			print('Processing Done.')
			hexcnt+=1
	print('\r', end='')
	print(' ' * 60, end='')
	print('\r\nFinished.')

if __name__ == '__main__':
	freeze_support()
	main()