import os
import glob
import h5py
import numpy as np
#import cv2
from util import *
import sys
'''
Note, these functions will not work without setting up the directories of video frames as shown in get_dir_lists. 
Alternatively, contact me to get access to the final h5Py datasets, which this code procudes.
'''

root_drive = '.' #Current dir for now

if not os.path.isdir(root_drive):
    print('Using Sharcnet equivalent of root_drive')
    root_drive = '/home/jjniatsl/project/jjniatsl/Fall-Data'

def get_dir_lists(dset):
    '''
    This shows structure which frames must be in
    
    Params:
        str dset: dataset to be loaded
    Returns:
        paths to ADL and Fall videos
    '''


    #----------USe these for N: drive located Fall-Data
    #root_drive = 'N:/FallDetection/Jacob/Fall-Data/' #Put Path to video frames 

    path_Fall = root_drive + '/Fall-Data/{}/Fall/Fall*'.format(dset)
    path_ADL = root_drive + '/Fall-Data/{}/NonFall/ADL*'.format(dset)
    if dset == 'Thermal-Dummy':

        path_Fall = root_drive + '/Fall-Data/Thermal-Dummy/Fall/Fall*'
        path_ADL = root_drive + '/Fall-Data/Thermal-Dummy/NonFall/ADL*'

    elif dset == 'Thermal':

        path_Fall = root_drive + '/Thermal/Fall/Fall*'
        path_ADL = root_drive + '/Thermal/NonFall/ADL*'
    
    elif dset == 'UR':
        path_Fall = root_drive + '/UR_Kinect/Fall/original/Fall*'
        path_ADL = root_drive + '/UR_Kinect/NonFall/original/adl*'
    
    elif dset == 'UR-Filled':
        path_Fall = root_drive + '/UR_Kinect/Fall/filled/Fall*'
        path_ADL = root_drive + '/UR_Kinect/NonFall/filled/adl*'

    elif dset == 'SDU':
        path_Fall = root_drive + '/SDUFall/Fall/Fall*/Depth'
        path_ADL = root_drive + '/SDUFall/NonFall/ADL*/Depth'
    
    elif dset == 'SDU-Filled':
        path_Fall = root_drive + '/SDUFall/Fall/Fall*/Filled'
        path_ADL = root_drive + '/SDUFall/NonFall/ADL*/Filled'
        
    print(path_Fall, path_ADL)
    vid_dir_list_Fall = glob.glob(path_Fall)
    vid_dir_list_ADL = glob.glob(path_ADL)

    if len(vid_dir_list_Fall) == 0:
        print('no Fall vids found')
    
    if len(vid_dir_list_ADL) == 0:
        print('no ADL vids found')

    return vid_dir_list_ADL, vid_dir_list_Fall


def init_videos(img_width = 64, img_height = 64, \
     raw = False, dset = 'Thermal'): 

    '''
    Creates or overwrites h5py group corresponding to root_path (in body), for the h5py file located at 
    'N:/FallDetection/Fall-Data/H5Data/Data_set-{}-imgdim{}x{}.h5'.format(dset, img_width, img_height). 

    The h5py group of nested groups is structured as follows:
    
    Processed (or Raw)
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

            ADL{N}
                Data
                    <HDF5 dataset "Data": shape (3203, 4096), type "<f8">
                Labels
                    <HDF5 dataset "Labels": shape (3203,), type "<i4">

            Fall1
                Data
                    <HDF5 dataset "Data": shape (49, 4096), type "<f8">
                Labels
                    <HDF5 dataset "Labels": shape (49,), type "<i4">
                .
                .
                .
            Fall{M}
                Data
                    <HDF5 dataset "Data": shape (49, 4096), type "<f8">
                Labels
                    <HDF5 dataset "Labels": shape (49,), type "<i4">


            where N is number of ADL videos, and M is number of Fall videos.

    Params:
        bool raw: if true, data will be not processed (mean centering and intensity scaling)
        int img_wdith: width of images
        int img_height: height of images
        str dset: dataset to be loaded

    '''
    path = root_drive + '/H5Data/{}/Data_set-{}-imgdim{}x{}.h5'.format(dset, dset, img_width, img_height) 

    vid_dir_list_0, vid_dir_list_1 = get_dir_lists(dset)

    if raw == False: 
        root_path = dset + '/Processed/Split_by_video'
    else:
        root_path = dset + '/Raw/Split_by_video'

    print('creating data at root_path', root_path)

    def init_videos_helper(root_path): #Nested to keep scope
            with h5py.File(path, 'a') as hf:
                #root_sub = root.create_group('Split_by_video')
                root = hf.create_group(root_path)

                for vid_dir in vid_dir_list_1:
                    init_vid(vid_dir = vid_dir, vid_class = 1, img_width = img_width, img_height = img_height,\
                     hf = root, raw = raw,  dset = dset)

                for vid_dir in vid_dir_list_0: 
                    init_vid(vid_dir = vid_dir, vid_class = 0, img_width = img_width, img_height = img_height, \
                        hf = root, raw = raw,  dset = dset)

    if os.path.isfile(path):
        hf = h5py.File(path, 'a')
        if root_path in hf:
            print('video h5py file exists, deleting old group {}, creating new'.format(root_path))
            del hf[root_path]
            hf.close()
            init_videos_helper(root_path)
        else:
            print('File exists, but no group for this data set; initializing..')
            hf.close()
            init_videos_helper(root_path)

    else:#not initialized
        print('No data file exists yet; initializing')

        init_videos_helper(root_path)


def init_vid(vid_dir = None, vid_class = None, img_width = 32, img_height = 32,\
     hf = None, raw = False,  dset = 'Thermal'):
    '''
    helper function for init_videos. Initialzies a single video.

    Params:
        str vid_dir: path to vid dir of frames to be initialzied
        int vid_class: 1 for Fall, 0 for NonFall
        h5py group: group within which new group is nested

    '''

    print('initializing vid at', vid_dir)

    data = create_img_data_set(fpath = vid_dir, ht = img_height, wd = img_width, raw = raw, sort = True, dset = dset)
    labels = np.zeros(len(data))

    if dset == 'SDU' or dset == 'SDU-Filled':
        vid_dir_name = os.path.basename(os.path.dirname(vid_dir))
    else:
        vid_dir_name = os.path.basename(vid_dir)
    print('vid_dir_name', vid_dir_name)
    grp = hf.create_group(vid_dir_name)



    if vid_dir_name in ['Fall' + str(i) for i in range(201)]: #201 is max fall index across all vids
        print('setting fall start')
        Fall_start, Fall_stop = get_fall_indeces(vid_dir_name, dset)
        labels[Fall_start:Fall_stop + 1] = 1
    
    grp['Labels'] = labels
    grp['Data'] = data

def get_fall_indeces(Fall_name, dset):
    root_dir = './Fall-Data/'
            
    labels_dir = root_dir + '/{}/Labels.csv'.format(dset)
    #print(labels_dir)
    import pandas as pd
    my_data = pd.read_csv(labels_dir, sep=',', header = 0, index_col = 0)
    
    start,stop = my_data.loc[Fall_name]
    print('start,stop', start,stop)
 
    #print(my_data)
    return start,stop
        

def sort_frames(frames, dset):
        #Sorting, trying for differnt dataset string formats
        if dset == 'SDU' or dset == 'SDU-Filled': #TODO remove try except, failing to sort shoudl stop!
            print('sorting SDU frames...')
            
            #try:
            frames = sorted(frames, key = lambda x: int(os.path.basename(x).split('.')[0])) #SDU
            # except ValueError:
            #     print('failed to sort SDU vid frames')
            #     pass
        elif dset == 'UR' or dset == 'UR-Filled' or dset == 'Thermal':
            print('sorting UR or Thermal frames...')
            try:
                frames = sorted(frames, key = lambda x: int(x.split('-')[-1].split('.')[0]))
            except ValueError:
                print('failed to sort UR vid frames')
                return
            
        elif dset == 'TST': 
            try:
                frames = sorted(frames, key = lambda x: int(x.split('_')[-1].split('.')[0]))
            except ValueError:
                print('failed to sort vid frames, trying again....')
                pass

        elif dset == 'FallFree' or dset == 'FallFree-Filled':
            try:
                frames = sorted(frames, key = lambda x: int(x.split('_')[2]))
            except ValueError:
                print('failed to sort vid frames, trying again....')
                pass

        return frames

def create_img_data_set(fpath, ht = 64, wd = 64, raw = False, sort = True, dset = 'Thermal'):
        '''
        Creates data set of all images located at fpath. Sorts images

        Params:
            str fpath: path to images to be processed
            bool raw: if True does mean centering and rescaling 
            bool sort: if True, sorts frames, ie. keeps sequential order, which may be lost due to glob
            dset: dataset

        Returns:
            ndarray data: Numpy array of images at fpath. Shape (samples, img_width*img_height),
            samples isnumber of images at fpath.

        '''
        
        #print('gathering data at', fpath)
        fpath = fpath.replace('\\', '/')
       # print(fpath+'/*.png')
        frames = glob.glob(fpath+'/*.jpg') + glob.glob(fpath+'/*.png')

        if sort == True:
            frames = sort_frames(frames, dset)

        #print("\n".join(frames)) #Use this to check if sorted

        data=np.zeros((frames.__len__(),ht,wd,1))
        for x,i in zip(frames, range(0,frames.__len__())):
            #print(x,i)
            img=cv2.imread(x,0) #Use this for RGB to GS
            #print('x', x)
            #img=cv2.imread(x,-1) #Use this for loading as is(ie. 16 bit needs this, else gets converted to 8)
           # print('img.shape', img.shape)
            img=cv2.resize(img,(ht,wd))#resize
            img=img.reshape(ht,wd,1)

            if raw == False:
                #print('proccessing data')

                img=img-np.mean(img)#Mean centering
                img=img.astype('float32') / 255. #rescaling

            data[i,:,:,:]=img

       # data = data.reshape((len(data), np.prod(data.shape[1:]))) #Flatten the images

        print('data.shape', data.shape)

        return data

def init_data_by_class(vid_class = 'NonFall', dset = 'Thermal',\
        raw = False, img_width = 64, img_height = 64, use_cropped = False): 

    '''
    Creates or overwrites h5py group corresponding to root_path (in body), for the h5py file located at 
    'N:/FallDetection/Fall-Data/H5Data/Data_set-{}-imgdim{}x{}.h5'.format(dset, img_width, img_height). 

    Creates the following structure:

    Processed
        Split_by_class
            NonFall
                Data
                    <HDF5 dataset "Data": shape (22116, 4096), type "<f8">
                Labels
                    <HDF5 dataset "Labels": shape (22116,), type "<i4">
            Fall
                Data
                    <HDF5 dataset "Data": shape (22116, 4096), type "<f8">
                Labels
                    <HDF5 dataset "Labels": shape (22116,), type "<i4">
    '''

    ht,wd = img_width, img_height
    if dset == 'Thermal':
        
        if vid_class == 'NonFall':
            fpath= root_drive + '/Thermal/{}/ADL*'.format(vid_class)            
        elif vid_class == 'Fall':
            fpath= root_drive + '/Thermal/{}/Fall*'.format(vid_class)           
        else:
            print('invalid vid class') 
            return


    elif dset == 'UR-Filled': 

        if vid_class == 'NonFall':
            fpath= root_drive + '/UR_Kinect/{}/filled/adl*'.format(vid_class)            
        else:
            fpath= root_drive + '/UR_Kinect/{}/filled/Fall*'.format(vid_class)            


    elif dset == 'UR':

        if vid_class == 'NonFall':
            fpath= root_drive + '/UR_Kinect/{}/original/adl*'.format(vid_class)            
        else:
            fpath= root_drive + '/UR_Kinect/{}/original/Fall*'.format(vid_class)            


    elif dset == 'SDU':
        fpath = root_drive + '/SDUFall/{}/ADL*/Depth'.format(vid_class)

    elif dset == 'SDU-Filled':
        fpath = root_drive + '/SDUFall/{}/ADL*/Filled'.format(vid_class)
        

    data = create_img_data_set(fpath, ht, wd, raw, False) #Don't need to sort

    #path = './H5Data/Data_set_imgdim{}x{}.h5'.format(img_width, img_height) #Old
    #path = 'N:/FallDetection/Fall-Data/H5Data/Data_set_imgdim{}x{}.h5'.format(img_width, img_height) #Old
    path = root_drive + '/H5Data/Data_set-{}-imgdim{}x{}.h5'.format(dset, img_width, img_height)

    if raw == False: 
        root_path = dset + '/Processed/Split_by_class/'+ vid_class #root path is for h5py tree         

    else:
        root_path = dset + '/Raw/Split_by_class/'+ vid_class

    if vid_class == 'NonFall':
        labels = np.array([0] * len(data))
    else:
        labels = np.array([1] * len(data))

     
    with h5py.File(path, 'a') as hf:
        #root_sub = root.create_group('Split_by_video')
        print('creating data at ', root_path)
        if root_path in hf:
            print('root_path {} found, clearing'.format(root_path))
            del hf[root_path]
        root = hf.create_group(root_path)

        root['Data'] = data

        root['Labels'] = labels

def flip_windowed_arr(windowed_data):
    """
    windowed_data: of shape (samples, win_len,...)
    
    returns shape len(windowed_data), win_len, flattened_dim)
    Note: Requires openCV
    """
    win_len = windowed_data.shape[1]
    flattened_dim = np.prod(windowed_data.shape[2:])
    #print(flattened_dim)
    flipped_data_windowed = np.zeros((len(windowed_data), win_len, flattened_dim)) #Array of windows
    print(flipped_data_windowed.shape)
    i=0
    for win_idx in range(len(windowed_data)):
        window = windowed_data[win_idx]
        flip_win = np.zeros((win_len, flattened_dim))

        for im_idx in range(len(window)):
            im = window[im_idx]
            hor_flip_im = cv2.flip(im,1)
            #print(hor_flip_im.shape)
            #print(flip_win[im_idx].shape)
            
            flip_win[im_idx] = hor_flip_im.reshape(flattened_dim)
            
        flipped_data_windowed[win_idx] = flip_win
    return flipped_data_windowed

