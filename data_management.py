import os
import glob
import h5py
import numpy as np
#import cv2
from util import *
import sys
root_drive = 'N:/FallDetection/Fall-Data/' 

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

    master_path = root_drive + '/H5Data/Data_set-{}-imgdim{}x{}.h5'.format(dset, img_width, img_height)

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
                save_path = root_drive + 'npData/{}/'.format(dset)

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
    '''

    vid_list = [len(vid['Data'][:]) for vid in list(vids_dict.values())]
    print(vid_list)

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
    To use this function, need to have downloaded h5py for dset, and placed in ./H5Data directory
    This is, aswell as sorting only function to be used from Git code
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
        print(root_path)

        with h5py.File(path, 'r') as hf:
            data_dict = hf[root_path]['Data'][:]
  
    return data_dict



