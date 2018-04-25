import numpy as np
from util import *
from data_management import *
from keras.models import load_model
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from read_activations import get_activations, display_activations
import h5py
import os
import glob
#import cv2


np.set_printoptions(threshold = np.nan)

class ImgExp:
	def __init__(self, model = None, img_width = None, img_height = None, model_name = 'None',\
		batch_size = 32, model_type = None, pre_load = None, initial_epoch = 0, epochs = 1, \
		zoom_range = 0, hor_flip = False, dset = 'Thermal'):
		'''
		Parent class for 
		'''
		
		self.dset = dset
		self.initial_epoch = initial_epoch
		self.model = model
		self.pre_load = pre_load
		self.img_width = img_width
		self.img_height = img_height
		self.model_name = model_name
		self.epochs = epochs
		
		self.batch_size = batch_size
		self.model_type = model_type
		self.zoom_range = zoom_range
		self.hor_flip = hor_flip
		
			
		if self.pre_load != None:
			print('loading saved model')
			self.model = load_model(pre_load)
			self.model_name = os.path.basename(pre_load).split('.')[0]

	def visualize(self):
		if self.model_type == 'conv':
			self.test_data = self.test_data[500].reshape(1,64,64,1)
		model, test_data = self.model, self.test_data
		maps = get_activations(model, test_data, True)
		display_activations(maps)
		return maps

	def save_exp(self):
	        
		#save_string = self.exp_name #Do this again incase info added to str based on data load(ie after init)
	    if self.hor_flip == True:
	        self.model_name = self.model_name + '-hor_flip'

	    base = './Models/{}/'.format(self.dset)

	    if not os.path.isdir(base):
	        os.makedirs(base)

	    save_string = '{}/{}.h5'.format(base, self.model_name)

	    print('saving model to ', save_string)
	    self.model.save(save_string)


	def load_train_data(self, raw = False): #TODO rename this function to load_train_data?
		raise NotImplementedError("Please Implement this method")

	def get_MSE(self, test_data):
		"""Returns MSE per input sample in test_data"""
		raise NotImplementedError("Please Implement this method")

	def train(self):
		raise NotImplementedError("Please Implement this method")
	def test(self):
		raise NotImplementedError("Please Implement this method")

	