import os
import glob
import h5py
import numpy as np
#import cv2
from util import *
import sys
from h5py_init import *
root_drive = '.' 

if not os.path.isdir(root_drive):
    print('Using Sharcnet equivalent of root_drive')
    root_drive = '/home/jjniatsl/project/jjniatsl/Fall-Data'


def init_windowed_arr(dset = 'Thermal', ADL_only = True, win_len = 8, img_width = 64, img_height = 64):
    '''
    Creates windowed version of dset data. Saves windowed array to 'npData/ADL_data-proc-win_{}.npy'.format(train_or_test, \
                    dset, win_len), vids_win)

    Params:
        str dset: dataset to use
        bool ADL_only: if True, only takes ADL from dataset
        int win_len: how many frames to extract for a sequence

    Returns:
        ndarray vids_win: shape (samples-D, win_len, )
    '''

    master_path = root_drive + '/H5Data/{}/Data_set-{}-imgdim{}x{}.h5'.format(dset,dset, img_width, img_height)

    if not os.path.isfile(master_path):
        print('initializing h5py..')
        init_videos(img_width = img_width, img_height = img_height, \
     raw = False, dset = dset)

    with h5py.File(master_path, 'r') as hf:

            data_dict = hf[dset + '/Processed/Split_by_video']

            if ADL_only == True:
                data_dict = dict((key,value) for\
                 key, value in data_dict.items() if 'adl' in key or 'ADL' in key) #Get only ADL vids

            vids_win = create_windowed_arr_per_vid(vids_dict = data_dict, \
                        stride = 1, \
                        win_len = win_len,\
                        img_width= img_width,\
                        img_height= img_height)

            if ADL_only == True:
                save_path = root_drive + '/npData/{}/'.format(dset)

                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
                save_path = save_path + 'ADL_data-proc-win_{}.npy'.format(win_len)

                print('saving data to ', save_path)
                np.save(save_path, vids_win)


            print('total windowed array shape', vids_win.shape)

    return vids_win

def create_windowed_arr_per_vid(vids_dict, stride, win_len, img_width, img_height):
    '''
    Assumes vids_dict is h5py structure, ie. vids_dict = hf['Data_2017/UR/Raw/Split_by_video']
    data set must cotnain atleast win_len frames
    '''

    vid_list = [len(vid['Data'][:]) for vid in list(vids_dict.values())]
    #print(vid_list)

    num_windowed = sum([int(np.floor(val-win_len)/stride)+1 for val in vid_list])
    #print('num_windowed', num_windowed)
    output_shape = (num_windowed, win_len,img_width, img_height, 1)
   # print('output_shape', output_shape)

    total = np.zeros(output_shape)
    #print('total.shape', 'num_windowed', 'output_shape', total.shape, num_windowed, output_shape)
    i=0
    for vid, name in zip(vids_dict.values(), vids_dict.keys()):
        print('windowing vid at', name)
        vid = vid['Data'][:]
        vid = vid.reshape(len(vid),64,64,1)
        vid_windowed = create_windowed_arr(vid, stride, win_len)
        print('windowed vid shape', vid_windowed.shape)
        total[i:i+len(vid_windowed)] = vid_windowed
        i += len(vid_windowed)

    return total


def create_windowed_arr(arr, stride, win_len):
    """
    arr: array of imgs
    """

    img_width, img_height = arr.shape[1], arr.shape[2]

    output_length = int(np.floor((len(arr) - win_len) / stride))+1
    output_shape = (output_length, win_len, img_width, img_height, 1)
    
    total = np.zeros(output_shape)
    
    i=0
    while i < output_length:
        next_chunk = np.array([arr[i+j] for j in range(win_len)]) #Can use np.arange if want to use time step \
        # ie. np.arrange(0,win_len,dt)

        total[i] = next_chunk
       
        i = i+stride

    arr_windowed = total

    return total



def load_data(split_by_vid_or_class = 'Split_by_vid', raw = False, img_width = 64, \
    img_height = 64, vid_class = 'NonFall', dset = 'Thermal'):
    """
    Note :to use this function, need to have downloaded h5py for dset, and placed in ./H5Data directory, or have downloaded data set,
    extracted frames, and placed them in directory structure specified in h5py_init.py
    
    Loads data from h5py file, and reutrns a dictionary, the properties of which depend on params vid_class and split_by_vid_or_class

    Params:
    	str split_by_vid_or_class: must be one of "Split_by_vid" or "Split_by_class". If "Split_by_vid", the returned dictionary
    	will have key-value pairs for each video. Otherwise, will have key-value paris for data and labels
    	bool raw: if true, data will be not processed (mean centering and intensity scaling)
    	int img_wdith: width of images
    	int img_height: height of images
        str dset: dataset to be loaded
    	str vid_class: must be one of "NonFall" or "Fall". if split_by_vid_or_class is "Split_by_class", will load only class
    		given by vid_class
    
    Returns:
    	h5py group data_dict: returns h5py nested group containing strucutred view of data. With 

					Split_by_class
						NonFall
							Data
								<HDF5 dataset "Data": shape (samples, img_height*img_width), type "<f8">
							Labels
								<HDF5 dataset "Labels": shape (samples,), type "<i4">

					Split_by_video
						ADL1
							Data
								<HDF5 dataset "Data": shape (1397, 4096), type "<f8">
							Labels
								<HDF5 dataset "Labels": shape (1397,), type "<i4">
						ADL2
							Data
								<HDF5 dataset "Data": shape (3203, 4096), type "<f8">
							Labels
								<HDF5 dataset "Labels": shape (3203,), type "<i4">

							.
							.
							.
						Fall1
							Data
								<HDF5 dataset "Data": shape (49, 4096), type "<f8">
							Labels
								<HDF5 dataset "Labels": shape (49,), type "<i4">
                            .
                            .
                            .


        See h5py_init documentation for more details on creation of the H5 Data.

    	
    """
    
    
    path = './H5Data/Data_set-{}-imgdim{}x{}.h5'.format(dset, img_width, img_height)

    path = 'N:/FallDetection/Fall-Data//H5Data/Data_set-{}-imgdim{}x{}.h5'.format(dset, img_width, img_height)#Local use only
    

    #init_h5py(path)

    if not os.path.isfile(path):
        print('h5py path {} not found'.format(path))
    
    else:
        print('h5py path found, loading data_dict..')
        if split_by_vid_or_class == 'Split_by_class':
            if raw == False: 
                root_path = dset + '/Processed/' + split_by_vid_or_class + '/' + vid_class
            else:
                root_path = dset + '/Raw/'+ split_by_vid_or_class + '/' + vid_class
        else:
            if raw == False: 
                root_path = dset + '/Processed/' + split_by_vid_or_class
            else:
                root_path = dset + '/Raw/'+ split_by_vid_or_class
        print('getting data at group', root_path)

        with h5py.File(path, 'r') as hf:
            data_dict = hf[root_path]['Data'][:]
  
    return data_dict



