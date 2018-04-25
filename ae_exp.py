from keras.models import Sequential, Model
from keras.layers import Activation, Dropout, Flatten, Dense, Input, Reshape
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from ImageExp import ImgExp
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
import time
from keras.models import load_model
import os
from util import *
from data_management import *


class AEExp(ImgExp):
	"""
	Frame based autoencoder experiment. Images are reconstructed individually, with no temporal information used.
	All params are attributes, and are initialized in ImgExp parent class.
	"""

	def __init__(self, model = None, img_width = None, img_height = None, \
		model_name = 'None', 
		batch_size = 32,\
		 model_type = None, \
		 pre_load = None, initial_epoch = 0, \
		epochs = 1, hor_flip = None,\
		 zoom_range = 0, dset = 'Thermal'):


		ImgExp.__init__(self, model = model, img_width = img_width,\
		 img_height = img_height, model_name = model_name, \
			batch_size = batch_size, \
			model_type = model_type, \
			pre_load = pre_load, initial_epoch = initial_epoch,\
			 epochs = epochs, hor_flip = hor_flip,\
		 	zoom_range = zoom_range, dset = dset)


	def play_frames_with_reconstructions(self, to_save = None):
		"""
		Plays frames of test_data with reconstuction.
		Params:
				bool to_save: if True, saves animation to 
		"""
		preds = self.model.predict(self.test_data.reshape(len(self.test_data),64,64,1))
		print(np.amax(preds[0]), np.amin(preds[0]))
		
		ani = animate_fall_detect(self.model, self.test_data, self.img_width, self.img_height)
		if to_save != None:
			ani.save('{}.mp4'.format(to_save))
		ani.event_source.stop()
		del ani
		plt.close()


	def train(self, sample_weight=None):
		"""
		trains the autoencoder model on data loaded from load_train_data. This data is non-sequential; that is,
		frames are reconstructed one by one. Reconstruction error (MSE) is minimized. Checkpoints and logs are saved to
		'./Checkpoints/dset/'
		'./logs/dset/'
		Model is saved as per save_exp method in parent class

		"""

		print(self.model.summary())
		model_name = self.model_name
		base_cp = './Checkpoints/{}'.format(self.dset)
		base_logs = './logs/{}'.format(self.dset)

		if not os.path.isdir(base_cp):
			os.makedirs(base_cp)
		if not os.path.isdir(base_logs):
			os.makedirs(base_logs)

		checkpointer = ModelCheckpoint( filepath = base_cp + '/' +  model_name + '-' + \
		'{epoch:03d}-{loss:.3f}.hdf5', period = 100, verbose =1)

		early_stopper = EarlyStopping(patience=5, verbose = 1, monitor = 'loss', min_delta = 1e-5)
		timestamp = time.time()
		csv_logger = CSVLogger(base_logs + '/' +  model_name + '-' + 'training-' + \
		    str(timestamp) + '.log')

		callbacks_list = [csv_logger, checkpointer]


		self.train_data = self.train_data.reshape(len(self.train_data) ,self.img_width, self.img_height, 1)
		
		datagen = ImageDataGenerator(horizontal_flip= self.hor_flip, zoom_range = self.zoom_range)


		print('training on data of shape {}, with model {}, with hor_flip {}'.format(self.train_data.shape, self.model_name, self.hor_flip))
		
		self.model.fit_generator(datagen.flow(self.train_data, self.train_data,\
		 batch_size = self.batch_size), steps_per_epoch=len(self.train_data) / self.batch_size, \
		epochs=self.epochs, callbacks = callbacks_list, verbose = 2)

		self.save_exp()


	def get_MSE(self, test_data):

		'''
		Gets mean squared error between test_data array of images, and their reconstructions from self.model

		Params:
				ndarray test_data: Data consiting of frames, ie. of dimension (samples, img_height, img_width)
		Returns:
				ndarray of MSE scores, one for each frame in test_data, ie dimensions (samples,1).

		'''

		if self.model_type == 'conv' or 1: #If using flow(no reason not to) then all take same shape TODO remove condition?
			test_data = test_data.reshape(len(test_data), self.img_width, self.img_height, 1)
		else:
			test_data = test_data.reshape((len(test_data), np.prod(test_data.shape[1:])))
		decoded_imgs = self.model.predict(test_data)


		RE = MSE(test_data, decoded_imgs)
		return RE

	def get_features(self, layer_name, train_or_test = 'test'):

		from keras.models import Model
		model = self.model  # create the original model 
		if train_or_test == 'test':
			data = self.test_data.reshape(len(self.test_data), self.img_width, self.img_height, 1)
						
		else:
			data = self.train_data.reshape(len(self.train_data), self.img_width, self.img_height, 1)

		#layer_name = 'my_layer' 
		intermediate_layer_model = Model(inputs=model.input,
		                           outputs=model.get_layer(layer_name).output)    
		intermediate_output = intermediate_layer_model.predict(data)
		#print(intermediate_output)
		return intermediate_output

	def load_train_data(self, raw = False): #TODO rename this function to load_train_data?
		"""
		"""
		split_by_vid_or_class = 'Split_by_class'
		vid_class = 'NonFall'

		data = load_data(split_by_vid_or_class = split_by_vid_or_class, raw = raw,\
		 img_width = self.img_width, img_height = self.img_height, vid_class = vid_class, dset = self.dset)
		
		self.train_data = data




