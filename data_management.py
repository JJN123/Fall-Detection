import os
import glob
import h5py
import numpy as np
#import cv2
from util import *
import sys

#TODO make this a class? This is data post proccessing, other in JNogasPy is data initialization


def create_windowed_arr(arr, stride, window_length):
    """
    arr: array of imgs
    """

    img_width, img_height = arr.shape[1], arr.shape[2]

    output_length = int(np.floor((len(arr) - window_length) / stride))+1
    output_shape = (output_length, window_length, img_width, img_height, 1)
    
    total = np.zeros(output_shape)
    
    i=0
    while i < output_length:
        next_chunk = np.array([arr[i+j] for j in range(window_length)]) #Can use np.arange if want to use time step \
        # ie. np.arrange(0,window_length,dt)

        total[i] = next_chunk
       
        i = i+stride

    arr_windowed = total

    return total



def load_data(split_by_vid_or_class = 'Split_by_vid', raw = False, img_width = 64, \
    img_height = 64, vid_class = 'NonFall', dset = 'Thermal'):
    """
    To use this function, need to have downloaded h5py for dset, and placed in ./H5Data directory
    This is, aswell as sorting only function to be used from Git code
    """
    
    
    path = './H5Data/Data_set-{}-imgdim{}x{}.h5'.format(dset, img_width, img_height)

    path = 'N:/FallDetection/Fall-Data//H5Data/Data_set-{}-imgdim{}x{}.h5'.format(dset, img_width, img_height)
    

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
        print(root_path)

        with h5py.File(path, 'r') as hf:
            data_dict = hf[root_path]['Data'][:]
  
    return data_dict



