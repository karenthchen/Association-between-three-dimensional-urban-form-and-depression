# @Date:   2019-05-13
# @Email:  thc@envs.au.dk Tzu-Hsin Karen Chen
# @Last modified time: 2020-10-07

import sys
#sys.path.insert(0, '/work/qiu/data4Keran/code/modelPredict')
sys.path.insert(0, '/home/xx02tmp/code3/modelPredict')
from img2mapC05 import img2mapC

import numpy as np
import time
sys.path.insert(0, '/home/xx02tmp/code3/dataPrepare')
import basic4dataPre
import h5py
import os
import glob2
import scipy.io as sio
from scipy import stats
import scipy.ndimage
import numpy.matlib
from numpy import argmax
from keras.utils import to_categorical
import skimage.measure

#image folder
imgFile_s2='/home/xx02tmp/image/to run49/'
#gt file folder
foldRef_LCZ=imgFile_s2

#class number
num_lcz=3
#stride to cut patches
step=24

patch_shape = (48, 48, 6)
#new line
img_shape = (48, 48)

#save folder
foldS='/home/xx02tmp/patch/patch50_11_02_48/'

params = {'dim_x': patch_shape[0],
		   'dim_y': patch_shape[1],
		   'dim_z': patch_shape[2],
		   'step': step,
		   'Bands': [0,1,2,3,4,5],
		   'scale':1.0,
		   'ratio':1,
		   'isSeg':0,
		   'nanValu':0,
		   'dim_x_img': img_shape[0],#the actuall extracted image patch
		   'dim_y_img': img_shape[1]}

#name of images
cities = ['summerrs2014_segA150sd']

#names of gt files
cities_ = ['class14_segA5530vp02n1_tra']

citiesval = ['summerrs2014_segA150sd']
cities_val =  ['class14_segA5530vp02n1_val']

#tra and vali patch numbers of each images
patchNum = np.zeros((2,len(cities)), dtype= np.int64) ;
#class number of each class
classNum = np.zeros((len(cities),3), dtype= np.int64) ; #change here

if not os.path.exists(foldS+'vali/'):
    os.makedirs(foldS+'vali/')
if not os.path.exists(foldS+'trai/'):
    os.makedirs(foldS+'trai/')

###########training patch#################
for idCity in np.arange(len(cities)):

	params['Bands'] = [0]
	params['scale'] = 1
	img2mapCLass=img2mapC(**params);

	###lcz to patches
	#load  file
	prj0, trans0, ref0= img2mapCLass.loadImgMat(foldRef_LCZ+cities_[idCity]+'.tif')
	print('ref0 size', ref0.shape)
	ref = np.int8(ref0)
	#print('lcz file size', ref.shape, trans0, ref.dtype)
	# to patches
	patchLCZ, R, C = img2mapCLass.label2patches_all(ref, 1)
	print('lcz patches, beginning', patchLCZ.shape, patchLCZ.dtype)

	#load img
	file =imgFile_s2 + cities[idCity] + '.tif'
	params['Bands'] = [0,1,2,3,4,5]
	params['scale'] = 1.0#!!!!!!!!!!!!!!!!!!!
	img2mapCLass=img2mapC(**params);
	prj0, trans0, img_= img2mapCLass.loadImgMat(file)
	print('img size', img_.shape)
	#image to patches
	patch_summer, R, C, idxNan = img2mapCLass.Bands2patches(img_, 1)
	print('image patches', patch_summer.shape, patch_summer.dtype)
        #try not delete idxNan (by Karen)
	print('lcz patches, before delete idxNan', patchLCZ.shape, patchLCZ.dtype)

	patchLCZ = np.delete(patchLCZ, idxNan, axis=0)
	print('lcz patches, after delete idxNan', patchLCZ.shape, patchLCZ.dtype)

	############manupulate the patches############
	#delete patches without lcz
	#change here, try 0.5
	c3Idx=basic4dataPre.patch2labelInx_lt(patchLCZ, 0, patchLCZ.shape[1],  patchLCZ.shape[2]*patchLCZ.shape[1]*0.044*1) 
	patchLCZ = np.delete(patchLCZ, c3Idx, axis=0)
	print('lcz patches, after delete noLCZ', patchLCZ.shape, patchLCZ.dtype)
	patch_summer = np.delete(patch_summer, c3Idx, axis=0)
	print('image patches, after delete noLCZ', patch_summer.shape, patch_summer.dtype)
	#print('delete no lcz patch: ', patchHSE.shape, patch_summer.shape, patchLCZ.shape)

	#NOT downsample to have a 90m gt
        #keep original 90m because of the new inputs of label has resoluiton at 90m
	#patchLCZ=skimage.measure.block_reduce(patchLCZ, (1,3,3,1), np.mean)
	patchLCZ=skimage.measure.block_reduce(patchLCZ, (1,1,1,1), np.mean)
	print('downsampled patchHSE:', patchLCZ.shape)

	###statistic of class number
	tmp=patchLCZ.reshape((-1,1))
	for c in np.arange(1,4): #change here class=1, 2, 3,4
		idx_=np.where(tmp==c)
		idx = idx_[0]
		classNum[idCity, c-1]=idx.shape[0]

	#reset the labels
	patchLCZ=patchLCZ-1; #0123, -1012
	#print('print(np.unique(patchHSE))',np.unique(patchLCZ))
	patchLCZ[patchLCZ==-1 ] = 3 #change here the low density class (0123)
	#patchLCZ=basic4dataPre.patchIndex2oneHot(patchLCZ, num_lcz)
	#print('final LCZ:', patchLCZ.shape, np.unique(patchLCZ))
	print('print(np.unique(patchLCZ))',np.unique(patchLCZ))
	print('shape', patchLCZ.shape, patch_summer.shape)

	patchNum_tra =basic4dataPre.savePatch_fold_single(patch_summer, patchLCZ, foldS+'trai/', cities[idCity])

	patchNum[0,idCity]=patchNum_tra
	#patchNum[1,idCity]=patchNum_val

	print(patchNum, classNum)


##############validation patch##############
print('start validation patch')
for idCity in np.arange(len(citiesval)):

	params['Bands'] = [0]
	params['scale'] = 1
	img2mapCLass=img2mapC(**params);

	###lcz to patches
	#load  file
	prj0, trans0, ref0= img2mapCLass.loadImgMat(foldRef_LCZ+cities_val[idCity]+'.tif')
	print('ref0 size', ref0.shape)
	ref = np.int8(ref0)
	#print('lcz file size', ref.shape, trans0, ref.dtype)
	# to patches
	patchLCZ, R, C = img2mapCLass.label2patches_all(ref, 1)
	print('lcz patches, beginning', patchLCZ.shape, patchLCZ.dtype)

	#load img
	file =imgFile_s2 + citiesval[idCity] + '.tif'
	params['Bands'] = [0,1,2,3,4,5]
	params['scale'] = 1.0#!!!!!!!!!!!!!!!!!!!
	img2mapCLass=img2mapC(**params);
	prj0, trans0, img_= img2mapCLass.loadImgMat(file)
	print('img size', img_.shape)
	#image to patches
	patch_summer, R, C, idxNan = img2mapCLass.Bands2patches(img_, 1)
	print('image patches', patch_summer.shape, patch_summer.dtype)
        #try not delete idxNan (by Karen)
	print('lcz patches, before delete idxNan', patchLCZ.shape, patchLCZ.dtype)

	patchLCZ = np.delete(patchLCZ, idxNan, axis=0)
	print('lcz patches, after delete idxNan', patchLCZ.shape, patchLCZ.dtype)

	############manupulate the patches############
	#delete patches without lcz
	#change here
	c3Idx=basic4dataPre.patch2labelInx_lt(patchLCZ, 0, patchLCZ.shape[1],  patchLCZ.shape[2]*patchLCZ.shape[1]*0.044*1)
	patchLCZ = np.delete(patchLCZ, c3Idx, axis=0)
	print('lcz patches, after delete noLCZ', patchLCZ.shape, patchLCZ.dtype)
	patch_summer = np.delete(patch_summer, c3Idx, axis=0)
	print('image patches, after delete noLCZ', patch_summer.shape, patch_summer.dtype)
	#print('delete no lcz patch: ', patchHSE.shape, patch_summer.shape, patchLCZ.shape)

	#NOT downsample to have a 90m gt
        #keep original 90m because of the new inputs of label has resoluiton at 90m
	#patchLCZ=skimage.measure.block_reduce(patchLCZ, (1,3,3,1), np.mean)
	patchLCZ=skimage.measure.block_reduce(patchLCZ, (1,1,1,1), np.mean)
	print('downsampled patchHSE:', patchLCZ.shape)

	###statistic of class number
	tmp=patchLCZ.reshape((-1,1))
	for c in np.arange(1,4): #change here
		idx_=np.where(tmp==c)
		idx = idx_[0]
		#classNum[idCity, c-1]=idx.shape[0]

	#reset the labels
	patchLCZ=patchLCZ-1;
	#print('print(np.unique(patchHSE))',np.unique(patchLCZ))
	patchLCZ[patchLCZ==-1 ] = 3 #change here
	#patchLCZ=basic4dataPre.patchIndex2oneHot(patchLCZ, num_lcz)
	#print('final LCZ:', patchLCZ.shape, np.unique(patchLCZ))
	print('print(np.unique(patchLCZ))',np.unique(patchLCZ))
	print('shape', patchLCZ.shape, patch_summer.shape)

	patchNum_val =basic4dataPre.savePatch_fold_singlev(patch_summer, patchLCZ, foldS+'vali/', cities[idCity])

	#patchNum[0,idCity]=patchNum_tra
	patchNum[1,idCity]=patchNum_val

	print(patchNum, classNum)

sio.savemat((foldS +'patchNum.mat'), {'patchNum': patchNum, 'classNum':classNum})
