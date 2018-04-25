from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, MaxPooling3D, UpSampling3D, Conv3D, Conv2DTranspose
from keras.layers import Activation, Dropout, Flatten, Dense, Input, Reshape

from keras.layers import Deconvolution3D
from keras.optimizers import SGD
from keras import regularizers
from keras.layers import LSTM, RepeatVector, concatenate
from keras import backend as K

"""
Defining Keras models as functions which return model object, aswell as model name and mode type strs.
All models take take img_width and img_height ints, which correpsond to dimensions of images passed to models.

"""


def DAE(img_width = 64, img_height = 64, regularizer_list = []):

	"""
	list regularizer_list: List of strings indicating which regulairzers to use, options are 
	'L1L2' and 'Dropout'. Can use both. Assume regularizer list ordered like ['L1L2', 'Dropout']
	"""
	encoding_dim = 500
	flatenned_dim = img_width*img_height
	input_shape = (img_width, img_height, 1)
	input_img_0 = Input(shape=input_shape)

	input_img = Reshape((flatenned_dim,), input_shape = input_shape)(input_img_0)

	if 'L1' in regularizer_list:
		activity_regularizer = regularizers.l1(0.1)
	else:
		activity_regularizer = None


	encoded = Dense(3*encoding_dim, activation='relu', activity_regularizer = activity_regularizer)(input_img)
	if 'Dropout' in regularizer_list:
		encoded = Dropout(0.25)(encoded)
	encoded = Dense(2*encoding_dim, activation='relu')(encoded)
	encoded = Dense(encoding_dim, activation='relu')(encoded)

	decoded = Dense(2*encoding_dim, activation='relu')(encoded)
	decoded = Dense(3*encoding_dim, activation='relu')(decoded)
	decoded = Dense(flatenned_dim, activation=output_activation)(decoded)
	decoded = Reshape((img_width, img_height, 1), input_shape = (flatenned_dim,))(decoded)

	autoencoder = Model(input_img_0, decoded)
	autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

	#-------------------------------------------------------
	
	encoded_input = Input(shape=(encoding_dim,))
	
	model = autoencoder

	model_name = 'DAE'

	for reg in regularizer_list:
		model_name += '-' + reg

	model_type = 'FC'

	return model, model_name, model_type

def CAE(img_width =64, img_height = 64, regularizer_list = []):
	"""
	list regularizer_list: List of strings indicating which regulairzers to use, options are 
	'L1L2' and 'Dropout'. Can use both. Assume regularizer list ordered like ['L1L2', 'Dropout']
	"""

	input_img = Input(shape=(img_width, img_height, 1))  # adapt this if using `channels_first` image data format

	if 'L1L2' in regularizer_list:
		kernel_regularizer=regularizers.l2(0.01)
		activity_regularizer=regularizers.l1(0.01)
	else:
		kernel_regularizer = None
		activity_regularizer = None

	x = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_regularizer=kernel_regularizer,\
		activity_regularizer = activity_regularizer)(input_img)
	x = MaxPooling2D((2, 2), padding='same')(x)

	if 'Dropout' in regularizer_list:
		x = Dropout(0.25)(x)

	x = Conv2D(8, (3, 3), activation='relu', padding='same', kernel_regularizer=kernel_regularizer,\
		activity_regularizer = activity_regularizer)(x)
	x = MaxPooling2D((2, 2), padding='same')(x)


	x = Conv2D(8, (3, 3), activation='relu', padding='same',kernel_regularizer=kernel_regularizer,\
		activity_regularizer = activity_regularizer)(x)
	encoded = MaxPooling2D((2, 2), padding='same')(x)


	x = Conv2DTranspose(8, (3, 3), activation='relu', padding='same')(encoded)
	x = UpSampling2D((2, 2))(x)
	x = Conv2DTranspose(8, (3, 3), activation='relu', padding='same')(x)
	x = UpSampling2D((2, 2))(x)
	x = Conv2DTranspose(16, (3, 3), activation='relu', padding = 'same')(x)
	x = UpSampling2D((2, 2))(x)
	decoded = Conv2DTranspose(1, (3, 3), activation='tanh', padding='same')(x)

	autoencoder = Model(input_img, decoded)
	autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')	

	model_name = 'CAE'
	for reg in regularizer_list:
		model_name += '-' + reg

	model_type = 'conv'

	model = autoencoder
	return model, model_name, model_type



def C3D_AE(img_width, img_height, win_length):
	"""
	int win_length: Length of window of frames
	"""

	input_shape = (win_length, img_width, img_height, 1)

	input_window = Input(shape = input_shape)

	temp_pool = 2
	temp_depth = 5
	x = Conv3D(16, (5, 3,3), activation='relu', padding='same')(input_window)
	x = MaxPooling3D((1,2, 2), padding='same')(x)

	x = Conv3D(8, (5, 3, 3), activation='relu', padding='same')(x)
	x = MaxPooling3D((temp_pool, 2, 2), padding='same')(x) #4
	x = Dropout(0.25)(x)
	x = Conv3D(8, (3, 3, 3), activation='relu', padding='same')(x)
	encoded = MaxPooling3D((temp_pool, 2, 2), padding='same')(x) #2

	# at this point the representation is (4, 4, 8) i.e. 128-dimensional

	x = Conv3D(8, (3, 3, 3), activation='relu', padding='same')(encoded)
	x = UpSampling3D((temp_pool, 2, 2))(x) #4
	x = Conv3D(8, (5, 3, 3), activation='relu', padding='same')(x)
	x = UpSampling3D((temp_pool, 2, 2))(x) #8

	x = Conv3D(16, (5, 3, 3), activation='relu', padding = 'same')(x)
	x = UpSampling3D((1, 2, 2))(x)
	decoded = Conv3D(1, (5, 3, 3), activation='tanh', padding='same')(x)



	autoencoder = Model(input_window, decoded)
	autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

	model_type = 'conv'
	model_name = 'C3D_AE'	
	encoder = None
	decoder = None
	model = autoencoder

	return model, model_name, model_type



def CAE3D(img_width, img_height, win_length):
    """
    int win_length: Length of window of frames
    """

    input_shape = (win_length, img_width, img_height, 1)

    input_window = Input(shape = input_shape)

    temp_pool = 2
    temp_depth = 5
    x = Conv3D(16, (temp_depth, 3,3), activation='relu', padding='same')(input_window)
    x = MaxPooling3D((2,2, 2), padding='same')(x)
    #x = Conv3D(8, (temp_depth, 3, 3), activation='relu', padding='same')(x)
    #x = MaxPooling3D((temp_pool, 2, 2), padding='same')(x)
    x = Dropout(0.25)(x)
    x = Conv3D(8, (temp_depth, 3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling3D((temp_pool, 2, 2), padding='same')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = Conv3D(8, (temp_depth, 3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling3D((temp_pool, 2, 2))(x)
    x = Conv3D(8, (temp_depth, 3, 3), activation='relu', padding='same')(x)
    x = UpSampling3D((temp_pool, 2, 2))(x)
    x = Conv3D(16, (temp_depth, 3, 3), activation='relu', padding = 'same')(x)
    #x = UpSampling3D((1, 2, 2))(x)
    decoded = Conv3D(1, (temp_depth, 3, 3), activation='tanh', padding='same')(x)

    autoencoder = Model(input_window, decoded)
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

    model_type = 'conv'
    model_name = 'CAE3D'	


    model = autoencoder

    return model, model_name, model_type

def CAE3D_deconv(img_width, img_height, win_length):
    """
    int win_length: Length of window of frames

    Replace Upsampling with Deconv
    """

    input_shape = (win_length, img_width, img_height, 1)

    input_window = Input(shape = input_shape)

    temp_pool = 2
    temp_depth = 5
    
    x = Conv3D(16, (temp_depth, 3,3), activation='relu', padding='same')(input_window)
    x = MaxPooling3D((2,2, 2), padding='same')(x)
    #x = Conv3D(8, (temp_depth, 3, 3), activation='relu', padding='same')(x)
    #x = MaxPooling3D((temp_pool, 2, 2), padding='same')(x)
    x = Dropout(0.25)(x)
    x = Conv3D(8, (temp_depth, 3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling3D((temp_pool, 2, 2), padding='same')(x)


    x = Deconvolution3D(8, (temp_depth, 3, 3),strides = (2,2,2), activation='relu', padding='same')(encoded)
    x = Deconvolution3D(16, (temp_depth, 3, 3),strides = (2,2,2), activation='relu', padding='same')(x)
    
    decoded = Conv3D(1, (temp_depth, 3, 3), activation='tanh', padding='same')(x)

    autoencoder = Model(input_window, decoded)
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

    model_type = 'conv'
    model_name = 'CAE3D_Deconv-pooling-win_{}'.format(win_length)	
    model = autoencoder

    return model, model_name, model_type

def CAE_deconv(img_width, img_height):
    """
    Replace Upsampling with Deconv
    """

    input_shape = (img_width, img_height, 1)

    input_frame = Input(shape = input_shape)
    
    x = Conv2D(16, (3,3), activation='relu', padding='same')(input_frame)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.25)(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2DTranspose(8, (3, 3),strides = (2,2), activation='relu', padding='same')(encoded)
    x = Conv2DTranspose(16, (3, 3),strides = (2,2), activation='relu', padding='same')(x)
    
    decoded = Conv2D(1, (3, 3), activation='tanh', padding='same')(x)

    autoencoder = Model(input_frame, decoded)
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

    model_type = 'conv'
    model_name = 'CAE_Deconv'
    model = autoencoder

    return model, model_name, model_type


def CLSTM_AE(img_width, img_height, win_len):

	"""
	from https://github.com/yshean/abnormal-spatiotemporal-ae/blob/master/classifier.py
	"""
	from keras.models import Model
	from keras.layers.convolutional import Conv2D, Conv2DTranspose
	from keras.layers.convolutional_recurrent import ConvLSTM2D
	from keras.layers.normalization import BatchNormalization
	from keras.layers.wrappers import TimeDistributed
	from keras.layers.core import Activation
	from keras.layers import Input

	input_tensor = Input(shape=(win_len, img_width, img_height, 1))

	conv1 = TimeDistributed(Conv2D(128, kernel_size=(11, 11), padding='same', strides=(4, 4), name='conv1'),
	                        input_shape=(win_len, 224, 224, 1))(input_tensor)
	conv1 = TimeDistributed(BatchNormalization())(conv1)
	conv1 = TimeDistributed(Activation('relu'))(conv1)

	conv2 = TimeDistributed(Conv2D(64, kernel_size=(5, 5), padding='same', strides=(2, 2), name='conv2'))(conv1)
	conv2 = TimeDistributed(BatchNormalization())(conv2)
	conv2 = TimeDistributed(Activation('relu'))(conv2)

	convlstm1 = ConvLSTM2D(64, kernel_size=(3, 3), padding='same', return_sequences=True, name='convlstm1')(conv2)
	convlstm2 = ConvLSTM2D(32, kernel_size=(3, 3), padding='same', return_sequences=True, name='convlstm2')(convlstm1)
	convlstm3 = ConvLSTM2D(64, kernel_size=(3, 3), padding='same', return_sequences=True, name='convlstm3')(convlstm2)

	deconv1 = TimeDistributed(Conv2DTranspose(128, kernel_size=(5, 5), padding='same', strides=(2, 2), name='deconv1'))(convlstm3)
	deconv1 = TimeDistributed(BatchNormalization())(deconv1)
	deconv1 = TimeDistributed(Activation('relu'))(deconv1)

	decoded = TimeDistributed(Conv2DTranspose(1, kernel_size=(11, 11), padding='same', strides=(4, 4), name='deconv2'))(
	    deconv1)

	model =  Model(inputs=input_tensor, outputs=decoded)
	model.compile(optimizer='adadelta', loss='mean_squared_error')

	model_name = 'CLSTM_AE'
	model_type = 'conv'

	return model, model_name, model_type



def dummy_3d(img_width, img_height, win_len):
	input_shape = (win_len, img_width, img_height, 1)

	input_window = Input(shape = input_shape)
	autoencoder = Model(input_window, input_window)
	autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

	model_type = 'conv'
	model_name = 'dummy_3d'	

	model = autoencoder

	return model, model_name, model_type

import numpy as np
if __name__ == "__main__":
	# model,_,_ = CLSTM_AE_tanh(64,64,8)
	# print(model.summary())
	# dummy = np.ones((1,8,64,64,1))*255
	# #dummy = dummy- np.mean(dummy)
	# pred = model.predict(dummy)
	# print(pred.shape)
	# print(np.amax(pred), np.amin(pred))
	pass
	# lstm,_,_ = Conv_LSTM_AE(28, 28, 2)
	# print(lstm.summary())
