# @Date:   2019-05-13
# @Email:  thc@envs.au.dk Tzu-Hsin Karen Chen
# @Last modified time: 2020-10-07

import sys
sys.path.insert(0, '/home/xx02tmp/code3/model')
sys.path.insert(0, '/home/xx02tmp/code3/dataPrepare')
import deepLabV3_ as DNN

import numpy as np
import time
from keras.models import load_model
import h5py
import os
import glob2
import scipy.io as sio
from img2mapC import img2mapC
from keras import backend as K
import os
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

#image patch
fileD='/home/xx02tmp/image/to run49/'
#model path
modelPath='/home/xx02tmp/results/50/deep2/m4_11_02_48_s2a124/'
nn=0# model id

patch_shape = (48, 48, 6)
step=patch_shape[0]
batchS=8#8

#saved weights
modelName =  modelPath +"0.0002_"+str(nn)+"_"+str(batchS)+"weights.best.hdf5"
#modelName =  modelPath +"0.0001_"+str(nn)+"_"+str(batchS)+"weights.best.hdf5"
#modelName =  modelPath +"epo_{epoch:02d}_0.0002_0_8model.final.hdf5"


###
params = {'dim_x': patch_shape[0],
		   'dim_y': patch_shape[1],
		   'dim_z': patch_shape[2],
		   'step': step,
		   'Bands': [0,1,2,3,4,5],#
		   'scale':1.0,
		   'isSeg':1, # if no downsampling to the image, the corner coordinates should kept the same
		   'nanValu':999,
		   'dim_x_img': patch_shape[0],#the actuall extracted image patch
		   'dim_y_img': patch_shape[1]}


cities = ['summerrs2014_segA150sd']



MapfileD=modelPath+'LczMap/'
if not os.path.exists(MapfileD):
	os.makedirs(MapfileD)

#for idCity in [0]:
for idCity in np.arange(len(cities)):

		if nn==0:#
			model = DNN.deepLabV3_(input_shape=patch_shape, out_shape=(48, 48), classes=3, midBlocks=4, atrous_rates = (1, 2, 4), entry_block3_stride = (1,1,2))
		if nn==1:#
			model = DNN.deepLabV3_(input_shape=patch_shape, out_shape=(48, 48), classes=3, midBlocks=4, atrous_rates = (1, 2, 4), entry_block3_stride = (1,1,3))

		model.load_weights(modelName, by_name=False)
		print(modelName)

		print(params['Bands'])
		img2mapCLass=img2mapC(**params);
		files=[fileD+cities[idCity]+'.tif']
		print(files)
		mapFile = MapfileD+cities[idCity]+'_'+ str(nn)+"_"+str(batchS)
		img2mapCLass.img2Bdetection_ovlp([files[0]], model, mapFile, out=1, nn=nn) #making prediction here
