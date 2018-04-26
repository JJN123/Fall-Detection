import numpy as np
from util import *
from keras.models import load_model
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import h5py
import os
import glob
#import cv2


np.set_printoptions(threshold = np.nan)

class ImgExp:

	"""
	Abstract parent class for ae_exp and seq_ae_exp. All params are attributes. Methods for training and testing are
	not implemented. Use children classes seq_ae_exp or ae_exp for training and testing. 
	ImgExp is short for image experiment. An experiment is an object which manages loading of image data, training an 
	autoencoder, aswell aswell as evaluating the trained model on a test set.

	"""
	def __init__(self, model = None, img_width = None, img_height = None, model_name = 'None',\
		batch_size = 32, model_type = None, pre_load = None, initial_epoch = 0, epochs = 1, \
		zoom_range = 0, hor_flip = False, dset = 'Thermal'):

		'''
		Args:
			model model: Keras model obect
			int img_width: width of images in experiment
			int img_height: height of images in experiment
			str model_name: name of model, used for saving model/model info.
			int batch_size: Number of samples in a batch
			str pre_load: path to model save
			int epochs: how many epochs to train for
			float zoom_range: as defined in Keras https://keras.io/preprocessing/image/
			bool hor_flip: if True then horiztonal flipping data augmentation is performed 
			(if sequence of images, then flips whole sequence)
			str dset: name of data set to be used in experiment. Current options are:
					Thermal: A dataset of thermal images
					UR-Filled: A dataset of kinect depth images with holes filled
					SDU-Filled: A dataset of kinect depth images with holes filled
			All datasets contain ADL in training data, and ADL + Falls in test data. 

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

	def save_exp(self):
	       
	    '''
	    Saves the model of the experiment to './Models/self.dset/self.model_name'
	    '''
		#save_string = self.exp_name #Do this again incase info added to str based on data load(ie after init)
	    if self.hor_flip == True:
	        self.model_name = self.model_name + '-hor_flip'

	    base = './Models/{}/'.format(self.dset)

	    if not os.path.isdir(base):
	        os.makedirs(base)

	    save_string = '{}/{}.h5'.format(base, self.model_name)

	    print('saving model to ', save_string)
	    self.model.save(save_string)


	def load_train_data(self, raw = False):
		"""
		Loads train data from memory. Train data will always consist solely of ADL images/image sequences.
		"""
		raise NotImplementedError("Please Implement this method")

	def get_MSE(self, test_data):
		"""Returns MSE per input sample in test_data"""
		raise NotImplementedError("Please Implement this method")

	def train(self):
		"""
		Train model
		"""
		raise NotImplementedError("Please Implement this method")
	def test(self):
		"""
		Evalute model based on reconstruciotn error as anomoly score for detecting falls in test video. Reports ROC AUC
		and PR AUC.
		"""
		raise NotImplementedError("Please Implement this method")

	