import os
import glob
import h5py
import numpy as np
#import cv2
from util import *
import sys
'''
Note, these functions will not work without setting up the directories of video frames as shown in get_dir_lists. Alternatively, contact me to get access to the final h5Py datasets, which this code procudes.
'''

root_drive = 'N:/FallDetection/Fall-Data/' #Put Path to video frames 

if not os.path.isdir(root_drive):
    print('Using Sharcnet equivalent of root_drive')
    root_drive = '/home/jjniatsl/project/jjniatsl/Fall-Data'

def get_dir_lists(dset, use_cropped = False):
    '''
    This shows structure which frames must be in
    '''

    if dset == 'Thermal':
        if use_cropped == True:
            
            path_NFFall = root_drive + '/DataforAE-json/test/NonFall/NFFall*/Cropped'
            path_Fall = root_drive + '/DataforAE-json/test/Fall/Fall*/Cropped'

            vid_dir_list_0 = glob.glob(path_NFFall) #NEeds more works if want seperated by vid ADL Cropped
            vid_dir_list_1 = glob.glob(path_Fall)             
        else:

            path_NFFall = root_drive + '/Thermal/DataforAE/test/NonFall/NFFall*'
            path_Fall = root_drive + '/Thermal/DataforAE/test/Fall/Fall*'
            path_ADL = root_drive + '/Thermal/DataforAE/train/NonFall/ADL*'

    elif dset == 'UR-Filled':
        path_NFFall = root_drive + '/UR_Kinect/test/NonFall/filled/NFFall*'
        path_ADL = root_drive + '/UR_Kinect/train/NonFall/filled/adl*'
        path_Fall = root_drive + '/UR_Kinect/test/Fall/filled/Fall*'

    elif dset == 'UR':
        path_NFFall = root_drive + '/UR_Kinect/test/NonFall/NFFall*'
        path_ADL = root_drive + '/UR_Kinect/train/NonFall/adl*'
        path_Fall = root_drive + '/UR_Kinect/test/Fall/Fall*'

    elif dset == 'TST':

        path_NFFall = root_drive + '/TST_Kinect_V2/Reorganized/test/NonFall/NFFall*'
        path_ADL = root_drive + '/TST_Kinect_V2/Reorganized/train/NonFall/ADL*'
        path_Fall = root_drive + '/TST_Kinect_V2/Reorganized/test/Fall/Fall*'


    elif dset == 'SDU':
        path_NFFall = root_drive + '/SDUFall/test/NonFall/NFFall*/Depth'
        path_ADL = root_drive + '/SDUFall/train/NonFall/ADL*/Depth'
        path_Fall = root_drive + '/SDUFall/test/Fall/Fall*/Depth'

    elif dset == 'SDU-Filled':
        path_NFFall = root_drive + '/SDUFall/test/NonFall/NFFall*/Filled'
        path_ADL = root_drive + '/SDUFall/train/NonFall/ADL*/Filled'
        path_Fall = root_drive + '/SDUFall/test/Fall/Fall*/Filled'

    vid_dir_list_0 = glob.glob(path_NFFall) + glob.glob(path_ADL)
    vid_dir_list_1 = glob.glob(path_Fall)

    return vid_dir_list_0, vid_dir_list_1

def init_videos(img_width = 64, img_height = 64, \
    use_cropped = False, raw = False, dset = 'Thermal'): 

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
            NFFall1
                Data
                    <HDF5 dataset "Data": shape (839, 4096), type "<f8">
                Labels
                    <HDF5 dataset "Labels": shape (839,), type "<i4">
                .
                .
                .
            Fall{M}
                Data
                    <HDF5 dataset "Data": shape (49, 4096), type "<f8">
                Labels
                    <HDF5 dataset "Labels": shape (49,), type "<i4">
            NFFall{M}
                Data
                    <HDF5 dataset "Data": shape (839, 4096), type "<f8">
                Labels
                    <HDF5 dataset "Labels": shape (839,), type "<i4">

            where N is number of ADL videos, and M is number of Fall videos. NFFall are the NonFall frames of a Fall video.

    Params:
        bool raw: if true, data will be not processed (mean centering and intensity scaling)
        int img_wdith: width of images
        int img_height: height of images
        str dset: dataset to be loaded


    '''
    path = root_drive + '/H5Data/Data_set-{}-imgdim{}x{}.h5'.format(dset, img_width, img_height) 
    

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
                     hf = root, raw = raw, use_cropped = use_cropped, dset = dset)

                for vid_dir in vid_dir_list_0: 
                    init_vid(vid_dir = vid_dir, vid_class = 0, img_width = img_width, img_height = img_height, \
                        hf = root, raw = raw, use_cropped = use_cropped, dset = dset)

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
     hf = None, raw = False, use_cropped = False, dset = 'Thermal'):
    '''
    helper function for init_videos. Initialzies a single video.

    Params:
        str vid_dir: path to vid dir of frames to be initialzied
        int vid_class: 1 for Fall, 0 for NonFall
        h5py group: group within which new group is nested

    '''

    print('initializing vid at', vid_dir)

    data = create_img_data_set(fpath = vid_dir, ht = img_height, wd = img_width, raw = raw, sort = True, dset = dset)
    labels = np.array([vid_class] * len(data))

    if dset == 'SDU' or dset == 'SDU-Filled':
        vid_dir_name = os.path.basename(os.path.dirname(vid_dir))
        
    else:
        vid_dir_name = os.path.basename(vid_dir)
    print('vid_dir_name', vid_dir_name)
    grp = hf.create_group(vid_dir_name)

    grp['Labels'] = labels
    grp['Data'] = data

    if vid_dir_name in ['Fall' + str(i) for i in range(201)]: #201 is max fall index across all vids
        Fall_start, Fall_stop = get_fall_indeces(vid_dir, use_cropped, dset)
        grp['Data'].attrs['Fall start index'] = Fall_start



def find_start_index_disc(start_frame_index, NF_frames_indeces):
    '''
    gets real index(not frame index) where Fall starts in NFFall array
    '''
    
    for i in range(len(NF_frames_indeces)):
        index = NF_frames_indeces[i]
        if index>start_frame_index:
            return i
        else:
            return start_frame_index

def get_fall_indeces(Fall_vid_dir = None, use_cropped = False, dset = 'Thermal'):
    """
    input Fall not NFFall (ie. recreate fall opt1, with opt3 labels)
    
    Gets start/stop indices accoutnign for potentiol discont's from cropping
    """
    if dset == 'Thermal' or dset == 'UR' or dset == 'UR-Filled':
        split_char = '-'
    else:
        split_char = '_'

    print('Fall_vid_dir', Fall_vid_dir)
    Fall_vid_dir = Fall_vid_dir.replace('\\','/')
    base_Fall = Fall_vid_dir
    
    basename = os.path.basename(Fall_vid_dir)
    print(basename)
    root = os.path.dirname(os.path.dirname(Fall_vid_dir))
    print(root)
    base_NFFall = root + '/NonFall/NF' + basename
        
    frames_opt3_Fall = glob.glob( base_Fall + '/*.jpg') + \
        glob.glob( base_Fall + '/*.png')
        
    frames_opt3_NFFall = glob.glob( base_NFFall + '/*.jpg') + \
        glob.glob( base_NFFall + '/*.png')
   
    
    #print("\n".join(frames_opt3_Fall))
    #print('{} opt3 fall frames found'.format(len(frames_opt3)))
    
    frames_opt3_Fall = sort_frames(frames_opt3_Fall, dset)
    frames_opt3_NFFall = sort_frames(frames_opt3_NFFall, dset)

    # if dset == 'TST': #Sortign frames, glob returns unsorted
    #     frames_opt3_Fall = sorted(frames_opt3_Fall, key = lambda x: int(x.split(split_char)[-1].split('.')[0]))


    frames_opt3_Fall[0] = frames_opt3_Fall[0].replace('\\', '/')
    print('frames_opt3_Fall[0]', frames_opt3_Fall[0])

    start_frame_ind = int(os.path.basename(frames_opt3_Fall[0]).split('.')[0].split(split_char)[-1])
    #start_frame_ind = int(frames_opt3_Fall[0].split(split_char)[-1].split('.')[0]) #Thermal
    print('start_frame_ind', start_frame_ind)
    end_frame_index = start_frame_ind + len(frames_opt3_Fall) #-1? TODO Not used 
    
    #print(frames_opt3_NFFall)
    NF_frame_indices = [int(os.path.basename(frames_opt3_NFFall[i]).split('.')[0].split(split_char)[-1]) \
                            for i in range(len(frames_opt3_NFFall))]
    
    if len(frames_opt3_NFFall) > 0:
        new_fall_start_ind = find_start_index_disc(start_frame_ind,\
                                                   NF_frame_indices)
    else:
        print('no NFF frames found')
        new_fall_start_ind = start_frame_ind
    
    print('new_fall_start_ind, len(frames_opt3_Fall)', new_fall_start_ind, len(frames_opt3_Fall))
    new_end_frame_index = new_fall_start_ind + len(frames_opt3_Fall)
    
    #New means accounts for discont of cropping
    
    return new_fall_start_ind, new_end_frame_index
        

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
        
        print('gathering data at', fpath)
        fpath = fpath.replace('\\', '/')
        frames = glob.glob(fpath+'/*.jpg') + glob.glob(fpath+'/*.png')

        if sort == True:
            frames = sort_frames(frames, dset)

       # print("\n".join(frames)) #Use this to check if sorted

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
                print('proccessing data')

                img=img-np.mean(img)#Mean centering
                img=img.astype('float32') / 255. #rescaling

            data[i,:,:,:]=img

        data = data.reshape((len(data), np.prod(data.shape[1:]))) #Flatten the images

        print('data.shape', data.shape)

        return data

def init_data_by_class(vid_class = 'NonFall', train_or_test = 'train', dset = 'Thermal',\
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
    if use_cropped == False:
        if dset == 'Thermal':
            if vid_class == 'NonFall':
                fpath= root_drive + '/Thermal/DataforAE' + \
                '/'+ train_or_test +'/' + vid_class + '/ADL*'               
            else:
                fpath= root_drive + '/DataforAE' + \
                '/'+ train_or_test +'/Fall/Fall*' 

        elif dset == 'UR-Filled': #TODO update!

            fpath = root_drive + '/UR_Kinect' + \
	            '/'+ train_or_test +'/' + vid_class + '/filled/adl*'

        elif dset == 'UR':
            fpath = root_drive + '/UR_Kinect' + \
	            '/'+ train_or_test +'/' + vid_class + '/adl*'

        elif dset == 'TST':
            fpath = root_drive + '/TST_Kinect_V2/Reorganized/train/NonFall/ADL*'

        elif dset == 'SDU':
            fpath = root_drive + '/SDUFall/{}/{}/ADL*/Depth'.format(train_or_test, vid_class)

        elif dset == 'SDU-Filled':
            fpath = root_drive + '/SDUFall/{}/{}/ADL*/Filled'.format(train_or_test, vid_class)


    elif vid_class == 'NonFall': #use cropped

        if dset == 'Thermal':
            fpath= root_Drive + '/DataforAE-json' + \
            '/'+ train_or_test +'/' + vid_class + '/' + 'Cropped'


    elif vid_class == 'Fall':#use cropped
        fpath= root_drive + '/DataforAE-json' + \
    '/'+ train_or_test +'/' + vid_class + '/Fall*/' + 'Cropped'

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

