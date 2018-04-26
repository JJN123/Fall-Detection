import os
import sys
from models import *
#from ImageExp import ImgExp
from ae_exp import AEExp
from util import *
import numpy as np
from keras.models import Sequential
import glob
from keras.models import load_model
import matplotlib.pyplot as plt

def init_dae_exp(pre_load = None, regularizer_list = []):
	batch_size = 16
		
	epochs = 1
	img_width, img_height = 64,64
	hor_flip = False

	misc_save_info =  None
	quick_test = False
	initial_epoch = 0
	

	dset = 'UR-Filled'

	#Deep_autoencoderflow------------

	autoencooder, model_name, model_type = DAE(img_width = img_width, 
		img_height = img_height, regularizer_list = regularizer_list)

	DAE_exp = AEExp(model = autoencooder, img_width = img_width,\
	img_height = img_height, model_name = model_name, model_type = model_type, \
	pre_load = pre_load, initial_epoch = initial_epoch,\
	epochs = epochs, batch_size = batch_size, dset = dset, hor_flip = hor_flip
	)

	return DAE_exp

if __name__ == "__main__":

	regularizer_list_list = [['Dropout']]

	for regularizer_list in regularizer_list_list:
		DAE_exp = init_dae_exp(regularizer_list = regularizer_list)

		DAE_exp.set_train_data(raw = False)
		print(DAE_exp.train_data.shape)

		DAE_exp.train()
		# #deepautoencoder_experiment.init_simple_encoder_decoder()

	

