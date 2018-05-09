from h5py_init import *


if __name__ == "__main__":

        # init_data_by_class(vid_class = 'NonFall', train_or_test = 'train', dset = 'SDU-Filled',\
        #       raw = False, img_width = 64, img_height = 64, use_cropped = False)


        init_videos(img_width = 64, img_height = 64, \
               use_cropped = False, raw = True, dset = 'Thermal')

        # init_videos(img_width = 64, img_height = 64, \
 #       use_cropped = False, raw = False, dset = 'UR-Filled', fill_holes = False)
