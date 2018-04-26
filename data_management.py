import os
import glob
import h5py
import numpy as np
#import cv2
from util import *
import sys

#TODO make this a class?

root_drive = 'N:/FallDetection/Fall-Data/' 

if not os.path.isdir(root_drive):
    print('Using Sharcnet equivalent of root_drive')
    root_drive = '/home/jjniatsl/project/jjniatsl/Fall-Data'

def get_dir_lists(dset, use_cropped = False):


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
    Creates or overwrites curent Data_set group corresponding to root_path. inits data by video, which is needed for temporal
    models
    '''
    #path = './H5Data/Data_set_imgdim{}x{}.h5'.format(img_width, img_height) #Old 

    #path = 'N:/FallDetection/Fall-Data/H5Data/Data_set_imgdim{}x{}.h5'.format(img_width, img_height)

    path = 'N:/FallDetection/Fall-Data/H5Data/Data_set-{}-imgdim{}x{}.h5'.format(dset, img_width, img_height)
    

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
     hf = None, raw = False, use_cropped = False, dset = 'Thermal'):# TOMOVE

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
        Creates data set of all images located at fpath. Sorts images (TODO make try catch for sorting?)
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

                img=img-np.mean(img)#Mean centering
                img=img.astype('float32') / 255. #normalize

            data[i,:,:,:]=img

        data = data.reshape((len(data), np.prod(data.shape[1:]))) #Flatten the images

        print('data.shape', data.shape)

        return data

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

def create_windowed_labels(labels, stride, tolerance, window_length):
    '''
    Create labels on seq level

    int tolerance: number of fall frames (1's) in a window for it to be labeled as a fall (1). must not exceed window length

    '''
    output_length = int(np.floor((len(labels) - window_length) / stride))+1
    #output_shape = (output_length, window_length, 1)
    output_shape = (output_length, 1)
    
    total = np.zeros(output_shape)
    
    i=0
    while i < output_length:
        next_chunk = np.array([labels[i+j] for j in range(window_length)])
        
        num_falls = sum(next_chunk) #number of falls in the window

        if num_falls >= tolerance:
            total[i] = 1
        else:
            total[i] = 0

       
        i = i+stride

    labels_windowed = total

    return labels_windowed

def create_windowed_arr_per_vid(vids_dict, stride, window_length, img_width, img_height):
    '''
    Assumes vids_dict is h5py structure, ie. vids_dict = hf['Data_2017/UR/Raw/Split_by_video']
    '''

        
    vid_list = [len(vid['Data'][:]) for vid in list(vids_dict.values())]
    print(vid_list)

    num_windowed = sum([int(np.floor(val-window_length)/stride)+1 for val in vid_list])
    print('num_windowed', num_windowed)
    output_shape = (num_windowed, window_length,img_width, img_height, 1)
    print('output_shape', output_shape)

    total = np.zeros(output_shape)
    print('total.shape', 'num_windowed', 'output_shape', total.shape, num_windowed, output_shape)
    i=0
    for vid, name in zip(vids_dict.values(), vids_dict.keys()):
        print('i',i, 'name', name)
        vid = vid['Data'][:]
        vid = vid.reshape(len(vid),64,64,1)
        vid_windowed = create_windowed_arr(vid, stride, window_length)
        print('vid_windowed.shape', vid_windowed.shape)
        total[i:i+len(vid_windowed)] = vid_windowed
        i += len(vid_windowed)
    print(total.shape)


    return total

def init_data_by_class(vid_class = 'NonFall', train_or_test = 'train', dset = 'Thermal',\
        raw = False, img_width = 64, img_height = 64, use_cropped = False, fill_holes = True): 

    '''
    Loads data seperated by vid class. To load data seperated by video, use init_videos
    Most usefull for 2D models(which do not require windowing by vid)

    Attributes:
        bool fill_holes: if True creates UR data set from denoised depth images
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


def load_data(split_by_vid_or_class = 'Split_by_vid', raw = False, img_width = 64, \
    img_height = 64, vid_class = 'NonFall', dset = 'Thermal'):

    path = 'N:/FallDetection/Fall-Data/H5Data/Data_set-{}-imgdim{}x{}.h5'.format(dset, img_width, img_height)

    #init_h5py(path)

    if not os.path.isfile(path):
        print('no h5py path found, initializing...')
        if split_by_vid_or_class == 'Split_by_class':
            init_data_by_class(vid_class = vid_class, img_width = 64, img_height = 64, \
                dset = dset, raw = raw)        

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
        try:
            with h5py.File(path, 'r') as hf:
                data_dict = hf[root_path]['Data'][:]
        except:
            print('component not found, initializing...')

            if split_by_vid_or_class == 'Split_by_class':
                init_data_by_class(vid_class = vid_class, img_width = 64, img_height = 64, \
                    dset = dset, raw = raw)
            else:
                init_videos(img_width = 64, img_height = 64, dset = dset, raw = raw)

        with h5py.File(path, 'r') as hf:
            data_dict = hf[root_path]['Data'][:]

    return data_dict


def create_windowed_train_data(window_length, dset = 'Thermal', raw = False, img_width = 64, img_height = 64):
    """
    Creates windowed data from ADL frames only
    TODO Delete? Repeat in data_man_main
    TODO MEM LEAK
    """

    #path = './H5Data/Data_set_imgdim64x64.h5'
    path = 'N:/FallDetection/Fall-Data/H5Data/Data_set_imgdim{}x{}.h5'.format(img_width, img_height)

    hf = h5py.File(path)
    if raw == False:
        data_dict = hf['Data_2017/'+ dset+'/Processed/Split_by_video']
    else:
        data_dict = hf['Data_2017/'+ dset+'/Raw/Split_by_video']
    data_dict_adl = dict((key,value) for key, value in data_dict.items() if 'adl' in key or 'ADL' in key)
    print(list(data_dict_adl.keys()))
    print(len(list(data_dict_adl.keys())))

    adl_win = create_windowed_arr_per_vid(vids_dict = data_dict_adl, \
                                stride = 1, \
                                window_length=window_length,\
                                img_width=img_width,\
                                img_height=img_height)

    print('adl_win.shape', adl_win.shape)
    np.save('./npData/train_data-'+dset+'-NonFalls-proc-windowed_by_vid-win_{}.npy'.format(window_length),adl_win)

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

