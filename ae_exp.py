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


	def display_reconstructions(self, fall_start = 0):
		
		if self.model_type == 'conv': #TODO shorten
			print('here')
			self.test_data = self.test_data.reshape(len(self.test_data), self.img_width, self.img_height, 1)
		else:
			self.test_data = self.test_data.reshape((len(self.test_data), np.prod(self.test_data.shape[1:])))
		
		fall_length = np.sum(self.test_labels) 
		num_NF = 20
		start_index = fall_start - num_NF
		dt = 4
		n = int(np.ceil((fall_length + num_NF)/dt))  # how many digits we will display

		decoded_imgs = self.model.predict(self.test_data)
	
		shown_test = []
		shown_decoded = []
		fig = plt.figure(figsize=(20, 4))
		fig.suptitle('Thermal Camera Image Reconstructions', fontsize = 16)
		for i in range(n): #3xn plot, at i+1th coutnign from left to right
			# display original
			ax = plt.subplot(3, n, i + 1)
			plt.imshow(self.test_data[start_index + dt*i].reshape(self.img_width, self.img_height), cmap = 'gray')
			#plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)

			if i == int(np.floor((n-1)/2)):
				ax.set_title('Original Frames', {'horizontalalignment' : 'left'})			

			# display reconstruction
			#plt.tight_layout()
			ax = plt.subplot(3, n, i + 1 + n)

			plt.imshow(decoded_imgs[start_index + dt*i].reshape(self.img_width, self.img_height), cmap = 'gray')
			
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)				
			if i == int(np.floor((n-1)/2)):
				ax.set_title('Reconstructed Frames',{'horizontalalignment' : 'left'})
			
#			plt.tight_layout()

			shown_test.append(self.test_data[start_index + dt*i].reshape(self.img_width, self.img_height))
			shown_decoded.append(decoded_imgs[start_index + dt*i].reshape(self.img_width, self.img_height))
			

		import matplotlib.gridspec as gridspec
		gs = gridspec.GridSpec(n,1)
		#gs.update(hspace = 0.000001)
		ax = plt.subplot(gs[-1, 0])

		#ax.set_ylim(0.005,0.03)
		ax.get_xaxis().set_visible(False)
		#ax.get_yaxis().set_visible(False)			
		#ax.set_ylabel('Reconstruction \n Error', rotation = 0, labelpad = 35)
		#ax.set_title('Reconstruction Error')
		
	
		plot_MSE_per_sample(np.array(shown_test), np.array(shown_decoded))
		#gs.tight_layout(fig, rect=[0, 0, 0.5, 1])
		#plt.subplots_adjust(hspace = 0.000000000000000000001)
		#plt.tight_layout()
		plt.show()
		return decoded_imgs

	def play_frames_with_reconstructions(self, to_save = None):
		preds = self.model.predict(self.test_data.reshape(len(self.test_data),64,64,1))
		print(np.amax(preds[0]), np.amin(preds[0]))
		# plt.imshow(preds[0].reshape(64,64))
		# plt.show()
		ani = animate_fall_detect(self.model, self.test_data, self.img_width, self.img_height)
		if to_save != None:
			ani.save('{}.mp4'.format(to_save))
		ani.event_source.stop()
		del ani
		plt.close()
		# decoded_imgs = self.model.predict()
		# play_frames(self.test_data, decoded_imgs)

	def train(self, sample_weight=None):
		"""
		trains the autoencoder model on data loaded from load_train_data. This data is non-sequential; that is,
		frames are reconstructed one by one. Reconstruction error (MSE) is minimized. Checkpoints and logs are saved to
		'./Checkpoints/dset/'
		'./logs/dset/'
		Model is saved as per save_exp method

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


	def get_MSE(self, test_data, test_vid = None):
		'''
		g
		'''

		if self.model_type == 'conv' or 1: #If using flow(no reason not to) then all take same shape TODO remove condition?
			test_data = test_data.reshape(len(test_data), self.img_width, self.img_height, 1)
		else:
			test_data = test_data.reshape((len(test_data), np.prod(test_data.shape[1:])))
		decoded_imgs = self.model.predict(test_data)
		#---
		# mn = np.mean(decoded_imgs, axis = (1,2,3))
		# print(mn.shape)
		# plt.plot(mn)
		# plt.show()


		RE = MSE(test_data, decoded_imgs)
		# print(RE.mean())
		# plt.plot(RE)
		# plt.plot(labels)
		# plt.show() 
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
		
		split_by_vid_or_class = 'Split_by_class'
		vid_class = 'NonFall'

		data = load_data(split_by_vid_or_class = split_by_vid_or_class, raw = raw,\
		 img_width = self.img_width, img_height = self.img_height, vid_class = vid_class, dset = self.dset)
		
		self.train_data = data




