import os
import time
import matplotlib
matplotlib.use('Agg')
from models import *
from util import *
from data_management import *
import numpy as np
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from keras.models import load_model
import h5py
import glob
from sklearn.metrics import average_precision_score
import sys
sys.path.insert(0,'./animation')
#from plot_video_animation_3D import * 
import pandas as pd
import matplotlib.pyplot as plt
from img_exp import ImgExp
root_drive = 'N:/FallDetection/Fall-Data/' 

if not os.path.isdir(root_drive):
    print('Using Sharcnet equivalent of root_drive')
    root_drive = '/home/jjniatsl/project/jjniatsl/Fall-Data/'

class SeqExp(ImgExp):
        '''
        A autoencoder experiment based on sequence of images
        Inherits get_thresh, save exp, and variable initialization

        Attributes:
            int win_len: window length, or the number of contigous frames forming a sample 
        '''
        def __init__(self, model = None, model_name = None, 
        misc_save_info = None, batch_size = 32, model_type = None, \
        callbacks_list = None, pre_load = None, initial_epoch = 0, epochs = 1, \
        dset = 'Thermal', win_len = 8, hor_flip = False, img_width = 64, img_height = 64):

            ImgExp.__init__(self, model = model, img_width = img_width,\
             img_height = img_height, model_name = model_name\
                , batch_size = batch_size, \
                model_type = model_type, \
                pre_load = pre_load, initial_epoch = initial_epoch,\
                 epochs = epochs, hor_flip = hor_flip,\
                 dset = dset)

            self.win_len = win_len

        def set_train_data(self, raw = False, mmap_mode = None):#TODO init windows from h5py if no npData found 
                '''
                loads or initazlzes windowed train data,  and sets self.train_data accordingly
                '''
                if self.dset == 'Thermal': #TODO rename npdata in accordance with new initialzer
                        to_load = root_drive + '/npData/train_data_NonFalls_proc_windowed_by_vid-win_{}.npy'.format(self.win_len)
                else:
                        to_load = root_drive + '/npData/{}/ADL_data-proc-win_{}.npy'.format(self.dset, self.win_len)
                
                if os.path.isfile(to_load):
                    print('npData found, loading..')
                    self.train_data = np.load(to_load, mmap_mode = mmap_mode)
                else:
                    print('npData not found, initializing..')

                    self.train_data = init_windowed_arr(dset = self.dset, ADL_only = True, win_len = self.win_len,
                        img_width = self.img_width, img_height = self.img_height)

                if self.hor_flip == True:

                        to_load_flip = './npData/hor_flip-by_window/{}'.format(os.path.basename(to_load))
                        data_flip = self.init_flipped_by_win(to_load_flip)

                        self.train_data = np.concatenate((self.train_data, data_flip), axis = 0)
                        
#                return self.train_data

        def train(self, sample_weight = None):
                """
                trains a sequential autoencoder on windowed data. That is, sequeneces of contigous frames
                are reconstucted.
                """

                model_name = self.model_name
                base = './Checkpoints/{}'.format(self.dset)
                if not os.path.isdir(base):
                    os.mkdir(base)

                checkpointer = ModelCheckpoint( filepath = base + '/' +  model_name + '-' + \
                    '{epoch:03d}-{loss:.3f}.hdf5', period = 100, verbose =1)
                timestamp = time.time()
                print('./Checkpoints/' + model_name + '-' + '.{epoch:03d}-{loss:.3f}.hdf5')
                csv_logger = CSVLogger('./logs/' + model_name + 'training-' + \
                str(timestamp) + '.log')

                #callbacks_list = [checkpointer, early_stopper, csv_logger]
                callbacks_list = [csv_logger, checkpointer]


                self.model.fit(self.train_data, self.train_data, epochs = self.epochs, batch_size = self.batch_size,\
                 verbose = 2, callbacks = callbacks_list, sample_weight = sample_weight)
                self.save_exp()

        def init_flipped_by_win(self, to_load_flip):

                if os.path.isfile(to_load_flip):
                        data_flip = np.load(to_load_flip)
                        data_flip = data_flip.reshape(len(data_flip), self.train_data.shape[1], self.train_data.shape[2],\
                        self.train_data.shape[3],1)

                        return data_flip
                        
                else:
                        print('creating flipped by window data..')
                        data_flip = flip_windowed_arr(self.train_data)

                        return data_flip



        def get_MSE(self, test_data, agg_type = 'x_std'):
            '''
            MSE for sequential data (video). Uses data chunking with memap for SDU-Filled. Assumes windowed
            
            Params:
                ndarray test_data: data used to test model (reconstrcut). Of 
                    shape (samples, window length, img_width, img_height)
                agg_type: how to aggregate windowde scores

            Returns:
                ndarray: Mean squared error between test_data windows and reconstructed windows, aggregated.
                This gives (samples,) shape
    
            '''

            import time

            
            img_width, img_height, win_len, model, stride = self.img_width, self.img_height, self.win_len, self.model, 1
            print('test_data.shape', test_data.shape)
            if test_data.shape[1] != win_len: #Not windowed
                test_data = test_data.reshape(len(test_data), img_width, img_height, 1)
                test_data = create_windowed_arr(test_data, stride, win_len)

            start_time = time.time()
            recons_seq = model.predict(test_data) #(samples-win_len+1, win_len, wd,ht,1)
            print(recons_seq.shape)
            elapsed_time = time.time() - start_time
            print('elapsed time for num frames', elapsed_time, len(test_data))
            recons_seq = recons_seq.reshape(len(recons_seq),win_len, img_height*img_width)#(samples-win_len+1, 5, wd*ht)
            test_data = test_data.reshape(len(test_data),win_len, img_height*img_width)#(samples-win_len+1, 5, wd*ht)

            RE = np.mean(np.power(test_data-recons_seq, 2), axis = 2)# (samples-win_len+1,win_len)

            RE = agg_window(RE, agg_type)

            return RE

        def get_MSE_all_agg(self, test_data):
            """
            Gets MSE for all aggregate types 'x_std', 'x_mean', 'in_std', 'in_mean'.

            Params:
                ndarray test_data: data used to test model (reconstrcut). Of 
                shape (samples, window length, img_width, img_height)

            Returns:
                dictionary with keys 'x_std', 'x_mean', 'in_std', 'in_mean', and values 
                ndarrays of shape (samples,)
            """

            img_width, img_height, win_len, model = self.img_width, self.img_height, self.win_len, self.model

            recons_seq = model.predict(test_data) #(samples-win_len+1, win_len, wd,ht,1)
            print(recons_seq.shape)

            recons_seq = recons_seq.reshape(len(recons_seq),win_len, img_height*img_width)#(samples-win_len+1, 5, wd*ht)
            test_data = test_data.reshape(len(test_data),win_len, img_height*img_width)#(samples-win_len+1, 5, wd*ht)

            RE = np.mean(np.power(test_data-recons_seq, 2), axis = 2)# (samples-win_len+1,win_len)
    
            RE_dict = {}

            agg_type_list = ['x_std', 'x_mean', 'in_std', 'in_mean']

            for agg_type in agg_type_list:

                RE_dict[agg_type] = agg_window(RE, agg_type)

            return RE_dict

        def test(self, animate = False):

            '''
            Gets AUC ROC/PR for all videos, using various (20) scoring schemes. Saves scores to 
            './AEComparisons/all_scores/self.dset/self.model_name.csv'

            Assumes self.model has been initialized
            '''

            dset, to_load, img_width, img_height = self.dset, self.pre_load, self.img_width, self.img_height
            stride = 1
            win_len = self.win_len


            model = self.model #TODO this can go in constructor
            model_name = os.path.basename(to_load).split('.')[0]

            print(model_name)
            print(model.summary())
            aucs = []
            std_total = []
            mean_total = []
            labels_total_l = []
            vid_index = 0 #vid index TODO rename

            vid_dir_keys_NFF = generate_vid_keys('NFFall', dset = dset) #ensures sorted order
            vid_dir_keys_Fall = generate_vid_keys('Fall', dset = dset)
            num_vids = len(vid_dir_keys_NFF)
            print('num_vids', num_vids)
            ROC_mat = np.ones((num_vids,18)) # 35 is num_vids, 20 scores-Xstd,Xmean,tols std..,tols mean..
            PR_mat = np.ones((num_vids,18))

            path = root_drive + 'H5Data/Data_set-{}-imgdim{}x{}.h5'.format(dset, img_width, img_height)

            with h5py.File(path, 'r') as hf:

                data_dict = hf['{}/Processed/Split_by_video'.format(dset)]
                
                for Fall_name, NFF_name in zip(vid_dir_keys_Fall, vid_dir_keys_NFF):

                    vid_total, labels_total = restore_Fall_vid(data_dict, Fall_name, NFF_name)

                    display_name = Fall_name

                    test_labels = labels_total

                    test_labels = labels_total
                    test_data = vid_total.reshape(len(vid_total), img_width, img_height, 1)
                    test_data_windowed = create_windowed_arr(test_data, stride, win_len)
                    
                    RE_dict = self.get_MSE_all_agg(test_data_windowed) #Return dict with value for each score style 

                    in_mean = RE_dict['in_mean']
                    in_std = RE_dict['in_std']

                    x_std = RE_dict['x_std']
                    x_mean = RE_dict['x_mean']
                    
                    
                    std_total.append(x_std)
                    mean_total.append(x_mean)
                    labels_total_l.append(labels_total)
                    inwin_labels = labels_total[win_len-1:]

                    auc_x_std, conf_mat, g_mean, ap_x_std = get_output(labels = test_labels,\
                            predictions = x_std, data_option = 'NA', to_plot = False)
                    auc_x_mean, conf_mat, g_mean, ap_x_mean = get_output(labels = test_labels,\
                            predictions = x_mean, data_option = 'NA', to_plot = False)

                    auc_in_std, conf_mat, g_mean, ap_in_std = get_output(labels = inwin_labels,\
                            predictions = in_std, data_option = 'NA', to_plot = False)
                    auc_in_mean, conf_mat, g_mean, ap_in_mean = get_output(labels = inwin_labels,\
                            predictions = in_mean, data_option = 'NA', to_plot = False)

                    ROC_mat[vid_index,0] = auc_x_std
                    ROC_mat[vid_index,1] = auc_x_mean
#                        ROC_mat[i,2] = auc_in_std
#                        ROC_mat[i,3] = auc_in_mean
                    tol_mat, tol_keys = gather_auc_avg_per_tol(in_mean, in_std, labels = test_labels, win_len = win_len)
                    AUROC_tol = tol_mat[0]
                    AUPR_tol = tol_mat[1]
                    num_scores_tol = tol_mat.shape[1]
                    print('num_scores_tol', num_scores_tol)
                    print('vid, auc_x_std', Fall_name, auc_x_std)

                    for k in range(num_scores_tol):
                        j = k+2 #start at 2, first two were for X_std and X_mean
                        ROC_mat[vid_index,j] = AUROC_tol[k]
                        PR_mat[vid_index,j] = AUPR_tol[k]

                    PR_mat[vid_index,0] = ap_x_std
                    PR_mat[vid_index,1] = ap_x_mean

                    vid_index += 1
                    
                    if animate == True:
                        ani_dir = './animation/{}/'.format(dset)
                        ani_dir = ani_dir + '/{}'.format(model_name)
                        if not os.path.isdir(ani_dir):
                            os.makedirs(ani_dir)

                        animate_fall_detect_Spresent(testfall = test_data, recons = recons_seq[:,4,:], \
                            scores = x_mean, to_save = ani_dir + '/{}.mp4'.format(Fall_name))


                #    break

                AUROC_avg = np.mean(ROC_mat, axis = 0)
                AUROC_std = np.std(ROC_mat, axis = 0)
                AUROC_avg_std = join_mean_std(AUROC_avg, AUROC_std)
                print(AUROC_std)
                AUPR_avg = np.mean(PR_mat, axis = 0)
                AUPR_std = np.std(PR_mat, axis = 0)

                AUPR_avg_std = join_mean_std(AUPR_avg, AUPR_std)
                total = np.vstack((AUROC_avg_std, AUPR_avg_std))
                
                df = pd.DataFrame(data = total, index = ['AUROC','AUPR'], columns = ['X-STD','X-Mean'] + tol_keys)
                

                base = './AEComparisons/all_scores/{}/'.format(self.dset)

                if not os.path.isdir(base):
                    os.mkdir(base)

                save_path = './AEComparisons/all_scores/{}/{}.csv'.format(dset, model_name)
                
                print(save_path)
                df.to_csv(save_path)
                






