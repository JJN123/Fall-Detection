from keras.models import load_model

from models import *
from util import *
import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd

import h5py
#import cv2
import sys
import pickle
from data_management import *
from sklearn.metrics import average_precision_score
from keras import backend as K

#Better name fot his file? Maybe just thresh tools?

root_drive = 'N:/FallDetection/Jacob/Fall-Data/' 

if not os.path.isdir(root_drive):
    print('Using Sharcnet equivalent of root_drive')
    root_drive = '/home/jjniatsl/project/jjniatsl/Fall-Data'



def get_stats_for_all_vids(experiment = None, thresholds = None, metric = 'G_Mean', models_dir = None,\
  dset = 'Thermal', agg_type = None, raw = False, animate= False):
	'''
	TODO auto initialize data if component not found etc.
	'''

		
	if thresholds != None:
		data_matrix = [['Mean Reconstruction Error', 'Mean Reconstruction Error + 1 :',\
		 'Mean Reconstruction Error + 2 :', 'Mean Reconstruction Error + 3 :', 'Maximum Reconstruction Error', 'RRE', 'ROC AUC Score', 'PR AUC Score']]
	else:
		data_matrix = [['ROC AUC Score', 'PR AUC Score']]


	vid_name_list = []
	features_list = []
	labels_list = []
	RE_l = []

	vid_dir_keys_Fall = generate_vid_keys('Fall', experiment.dset) #Ensures sorted order of fall vids
	
	path = root_drive + '/H5Data/Data_set-{}-imgdim{}x{}.h5'.format(experiment.dset, experiment.img_width, experiment.img_height)

	Fall_stop = 'None' #Make th

	with h5py.File(path, 'r') as hf:
		data_dict = hf[dset + '/Processed/Split_by_video']
		f_idx=0
		for Fall_name in vid_dir_keys_Fall:

			if Fall_name == Fall_stop:
				print('breaking at ', Fall_name)
				break

			vid_total = data_dict[Fall_name]['Data'][:]
			labels_total = data_dict[Fall_name]['Labels'][:]

			experiment.test_data = vid_total
			display_name = Fall_name
			vid_name_list.append(display_name)
			print('testing on', display_name)

			next_row, RE = get_stats_for_vid(test_data = vid_total, test_labels = labels_total, \
				experiment = experiment, thresholds = thresholds, metric = metric, \
				Fall_name = Fall_name, dset = dset, agg_type = agg_type, f_idx = f_idx)
			f_idx+=1

			RE_l.append(RE)

			data_matrix.append(next_row)


			if animate == True:
				ani_dir = './animation/{}/'.format(dset)
				ani_dir = ani_dir + '/{}'.format(experiment.model_name)
				if not os.path.isdir(ani_dir):
					os.makedirs(ani_dir)
				preds = experiment.model.predict(vid_total.reshape(len(vid_total), experiment.img_width, experiment.img_height, 1))
				animate_fall_detect_Spresent(testfall = vid_total, recons = preds, scores = RE, to_save = ani_dir + '/{}.mp4'.format(Fall_name))

	headers = data_matrix.pop(0)
	print(headers)
	print(np.array(data_matrix).shape)
	df = pd.DataFrame(data_matrix, index = vid_name_list, columns=headers)
	df.loc['mean'] = df.mean()

	df.loc['std'] = df[0:-1].std()
	df = df.round(2)


	if agg_type != None:
	    root_s = './AEComparisons/' + metric + '/' + experiment.dset + '/' + agg_type
	else:
	    root_s = './AEComparisons/' + metric + '/' + experiment.dset + '/'

	if not os.path.isdir(root_s):
	    os.makedirs(root_s)
	save_path = root_s + '/' + experiment.model_name + '.csv'

	print('saving results to', save_path)
	df.to_csv(save_path)
	#print(df)
		

def get_stats_for_vid(test_data = None, experiment = None, thresholds = None, metric = None, \
	test_labels = None, Fall_name = None, dset = None, agg_type = None, f_idx = None):
		
	if agg_type != None: #Must window if agg_type != None
		print('windowing data')
		img_width, img_height, win_len, stride = experiment.img_width, experiment.img_height, experiment.win_len, 1
		test_data = test_data.reshape(len(test_data), img_width, img_height, 1)
		test_data = create_windowed_arr(test_data, stride, win_len)

	labels = test_labels
	RE = experiment.get_MSE(test_data)


	#print('np.amin(RE), np.amax(RE)', np.amin(RE), np.amax(RE))
	next_row = []
	
	if thresholds != None:
		for t in thresholds:
			AUROC, conf_mat, g_mean, AUPR = get_output(labels, RE, 'AE', t=t, to_plot = False)
			tn, fp, fn, tp = conf_mat.ravel()

			FPR = fp/(fp+tn)
			TPR = tp/(tp+fn)
			
			if metric == 'G_Mean':
				next_row.append(g_mean)
			elif metric == 'FPR':
				next_row.append(FPR)
			elif metric == 'TPR':
				next_row.append(TPR)
	else:
		AUROC, conf_mat, g_mean, AUPR = get_output(labels, RE, 'AE', to_plot = False)


	print('AUROC', AUROC)
	print('AUPR', AUPR)
	next_row.append(AUROC)
	next_row.append(AUPR)
	#print(next_row)

	return next_row, RE


def create_all_pds(experiments, AUC_only = False, models_dir = None, dset = 'Thermal', agg_type = None):
	'''
	agg_type is method of aggregating window of scores
	'''

	if AUC_only == True:
		metrics = ['ROC_AUC']

	else:
		metrics = ['G_Mean']
		metrics = ['G_Mean', 'TPR', 'FPR']

	for exp in experiments:
		if AUC_only == False:
		    
		    #load thresholds
		    #if not there..init
		    if exp.dset == 'SDU-Filled' or exp.dset == 'SDU':
		        exp.load_train_data(raw = False, mmap_mode = 'r')
		        thresholds = exp.get_thresholds(train_data = exp.train_data, agg_type = agg_type)

		    else:
		        exp.load_train_data(raw = False)

		        thresholds = exp.get_thresholds(train_data = exp.train_data, agg_type = agg_type)
		    print('got thresholds')
		    #train_data = None
		    exp.train_data = None

		else:
		    thresholds = None
		for metric in metrics:
			get_stats_for_all_vids(experiment = exp, thresholds = thresholds,\
			metric = metric, models_dir = models_dir, dset = dset, agg_type = agg_type)



	 
def save_features(features, model_name, classes, train_or_test, non_zero_idxs):
	'''
	non_zero_idxs must come from train set
	'''
	import pandas as pd 
	import csv
	#chararr = np.chararray(len(features))
	chararr = []
	header = []
	features = features[:,non_zero_idxs]

	for i in range(features.shape[1]):#for each col
		header.append('feature_' + str(i))


	df = pd.DataFrame(data = features, columns = header)
	df['class'] = classes
	
	df.to_csv("./features/Thermal/{}_data-{}.csv".format(train_or_test, model_name), index=False, quoting=csv.QUOTE_NONE)

def gather_and_save_feautres(experiment, dset):
	'''
	Gets train and test features, also gets non zero idx's form train set.
	'''
	#Getting/Saving train features-----------------------
	cae_exp.init_data(raw = False, split_by_vid_or_class = 'Split_by_class', vid_class = 'NonFall')
	layer_name = 'max_pooling2d_3'

	features = cae_exp.get_features(layer_name = layer_name, train_or_test = 'train')

	print(features.shape)

	features = features.reshape(len(features), np.prod(features.shape[1:]))
	#features = features[0:856,0:20]
	non_zero_idxs = ~(features == 0).all(axis=0)
	
	classes = []
	for j in range(features.shape[0]):
		classes.append('n')
	train_or_test = 'train'
	save_features(features, cae_exp.model_name, classes, train_or_test, non_zero_idxs)

	#---------------------------------
	features_list, classes = gather_test_features(experiment, dset)
	train_or_test = 'test'
	save_features(features_list, experiment.model_name, classes, train_or_test, non_zero_idxs)

def gather_test_features(experiment = None, dset = 'Thermal'):	
	
	vid_name_list = []
	features_list = []
	labels_list = []
	vid_dir_keys_NFF = generate_vid_keys('NFFall', experiment.dset) #ensures sorted order
	vid_dir_keys_Fall = generate_vid_keys('Fall', experiment.dset)

	path = 'N:/FallDetection/Fall-Data/H5Data/Data_set_imgdim{}x{}.h5'.format(experiment.img_width, experiment.img_height)

	Fall_stop = 'None'
	with h5py.File(path, 'r') as hf:
		#train_dict = hf['Data_2017/Thermal/Raw/Split_by_class'] 

		if dset != 'UR':
				data_dict = hf['Data_2017/' + dset + '/Processed/Split_by_video']
		else:
				data_dict = hf['Data_2017/UR/Processed/Filled/Split_by_video']
				data_dict = hf['Data_2017/' + dset + '/Processed/Split_by_video']

		RE_old = 0
		for Fall_name, NFF_name in zip(vid_dir_keys_Fall, vid_dir_keys_NFF):

			if Fall_name == Fall_stop:
				print('breaking at ', Fall_name)
				break

			fall_start = data_dict[Fall_name + '/Data'].attrs['Fall start index'] #Restores sequence order, experiment.use_cropped != data.use_cropped always
			if dset == 'UR' or dset == 'UR-Filled':
				fall_start -= 1

			Fall_data, Fall_labels = data_dict[Fall_name + '/Data'][:], data_dict[Fall_name + '/Labels'][:]
			NFF_data, NFF_labels = data_dict[NFF_name+ '/Data'][:], data_dict[NFF_name+ '/Labels'][:]
			vid_total = np.concatenate((NFF_data[:fall_start], Fall_data, NFF_data[fall_start:]),axis=0)
			labels_total = np.concatenate((NFF_labels[:fall_start], Fall_labels, NFF_labels[fall_start:]),axis=0)
			
			experiment.test_data = vid_total

			#Saving Test Features-----------------------------
			layer_name = 'max_pooling2d_3'
			# #experiment.play_frames_with_reconstructions()
			features = experiment.get_features(layer_name)
			features_list.append(features)
			#labels_list.append(labels_total)

			for i in range(len(labels_total)):
				lab = labels_total[i]
				if lab == 1:
					labels_list.append('o')
				else:
					labels_list.append('n')


			# print('features.shape', features.shape)
			# print(np.amax(features), np.amin(features))
			# #print(features[0])
			# plt.figure()

			# for i in range(7):
			# 	i+=1
			# 	ax = plt.subplot(1,8,i)
			# 	plt.imshow(features[300,:,:,0], cmap = 'gray')
			# plt.show()
			# break
			#-----------------------------
			display_name = Fall_name
			print('testing on', display_name)



	features_list = np.vstack(features_list)
	print('features_list.shape', features_list.shape)
	features_list = features_list.reshape(len(features_list), np.prod(features_list.shape[1:]))
	print('features_list.shape', features_list.shape)
	#features_list = features_list[:,~(features_list == 0).all(axis=0)]


	classes = labels_list
	
	#If truncating for testing----
	#features_list = features_list[0:856,0:20]
	#classes = classes[0:856]
	return features_list, classes
