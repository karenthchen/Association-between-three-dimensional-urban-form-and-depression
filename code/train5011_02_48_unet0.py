# @Date:   2019-05-13
# @Email:  thc@envs.au.dk Tzu-Hsin Karen Chen
# @Last modified time: 2020-10-07


###
import sys
sys.path.insert(0, '/home/xx02tmp/code3/dataPrepare')
from dataGener import DataGenerator
import utli

sys.path.insert(0, '/home/xx02tmp/code3/model')
import deepLabV3_ as DNN

from keras.callbacks import TensorBoard

from pathlib import Path
import h5py
import numpy as np
from random import shuffle
import pickle
import glob, glob2
import time
import scipy.io as sio

from keras.utils.vis_utils import plot_model
from keras.optimizers import Nadam
from keras.optimizers import SGD
#from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
#from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
from batch_logger import NBatchLogger


import tensorflow as tf
#problem: out of memory
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from keras import backend as K
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.9#0.41
#session = tf.Session(config=config)

# sys.path.insert(0, '/home/qiu/CodeSummary/LCZ_CPQiu_util/lr')
# from LRFinder import LRFinder
#################################################################
#changeable para
patch_shape = (48, 48, 6)
classNumber=3
#patch folder
folderData = '/home/xx02tmp/patch/patch50_11_02_48/'
#results folder
file0='/home/xx02tmp/results/50/unet0/m4_11_02_48_s2a124/'

nn = 5

#epochs = 45
epochs = 10
batch_size = 8#8
learnRate = 0.0002#0.0002
nadam = Nadam(lr = learnRate)

timeCreated =time.strftime('%Y-%m-%d-%H', time.localtime(time.time()))
fileSave=  file0 + str(learnRate) + '_' + str(nn) + '_' + str(batch_size)
checkpoint = ModelCheckpoint(fileSave+"weights.best.hdf5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

tbCallBack = TensorBoard(log_dir=file0+'logs' + '_' + str(nn) + '_' + str(batch_size) + '_' + timeCreated,  # log 目录
                 histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
#                  batch_size=32,     # 用多大量的数据计算直方图
                 write_graph=True,  # 是否存储网络结构图
                 write_grads=True, # 是否可视化梯度直方图
                 write_images=True,# 是否可视化参数
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None)
###################################################################
if not os.path.exists(file0):
	os.makedirs(file0)

#number of training and validate samples
mat = sio.loadmat(folderData+'patchNum.mat')
patchNum=mat['patchNum']*1

traNum=np.sum(patchNum[0,:]) ;
valNum=np.sum(patchNum[1,:]) ;
print('train val:', traNum, valNum)

#validata samples
fileVal = glob2.glob(folderData+'vali/' +'*.h5')

#training samples
fileTra = glob2.glob(folderData+'trai/' +'*.h5')
shuffle(fileVal)
shuffle(fileTra)
print('train files:', len(fileTra))
print('vali files:',  len(fileVal))
print('nn',nn)
# Generators
params = {'dim_x': patch_shape[0],
		  'dim_y': patch_shape[1],
		  'dim_z': patch_shape[2],
		  'batch_size':batch_size,
          'band': [0,1,2,3,4,5],
		  'flow': 0}
tra_generator = DataGenerator(**params).generate(fileTra)
val_generator = DataGenerator(**params).generate(fileVal)


# if nn==0:#
# 	model = DNN.deepLabV3_(input_shape=patch_shape, out_shape=(48, 48), classes=classNumber, midBlocks=4, atrous_rates = (1, 2, 4), entry_block3_stride = (1,1,2))
# if nn==1:#
# 	model = DNN.deepLabV3_(input_shape=patch_shape, out_shape=(48, 48), classes=classNumber, midBlocks=4, atrous_rates = (1, 2, 4), entry_block3_stride = (1,1,3))


if nn==3:#
	import fcn_net
	model = fcn_net.sen2IS_net_bn(input_size = patch_shape, numC=classNumber)


if nn==5:#
	import uNet_0
	model = uNet_0.unet(input_size = patch_shape, numC=classNumber)
	#model = u_net.unet(input_size = patch_shape, numC=classNumber)#output 24

model.compile(optimizer=nadam, loss=utli.masked_loss_function, metrics=['accuracy'])

start = time.time()
model.fit_generator(generator = tra_generator,
				steps_per_epoch = traNum//batch_size, epochs = epochs,
				validation_data = val_generator,
				validation_steps = valNum//batch_size,
				callbacks = [checkpoint, tbCallBack], max_queue_size = 100, verbose=1)#
end =time.time()

plot_model(model, to_file=file0+'model_plot.png', show_shapes=True, show_layer_names=True)

with open(file0 + 'report.txt','w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: fh.write(x + '\n'))
			

trainingTime=end-start;
sio.savemat((fileSave+'_trainingTime.mat'), {'trainingTime':trainingTime})
