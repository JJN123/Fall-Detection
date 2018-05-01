from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, f1_score, auc, precision_recall_curve
import glob
import os
import numpy as np

#from pathlib import Path
from sklearn.utils import class_weight as cw
#import h5py
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils.io_utils import HDF5Matrix
import matplotlib.pyplot as plt
import cv2
import h5py
from random import randint, seed
from imblearn.metrics import geometric_mean_score
import re
import sys
#from data_management import *


def threshold(predictions = None, t = 0.5):
        temp = predictions.copy()
        predicted_classes = temp.reshape(predictions.shape[0])
        for i in range(len(predicted_classes)):
            if predicted_classes[i]<t:
                predicted_classes[i] = 0
            else:
                predicted_classes[i] = 1

        return predicted_classes

import pandas as pd



def plot_ROC_AUC(fpr, tpr, roc_auc, data_option):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for {}'.format(data_option))
    plt.legend(loc="lower right")
    plt.show()

def get_output(labels, predictions, data_option = None, t=0.5, to_plot = False, pos_label = 1):
    predicted_classes = threshold(predictions, t)
    true_classes = labels
    conf_mat = confusion_matrix(y_true = true_classes, y_pred = predicted_classes)
    #report = classification_report(true_classes, predicted_classes)
    AUROC = []
    AUPR = []
    if np.count_nonzero(labels) > 0 and np.count_nonzero(labels) != labels.shape[0]: #Makes sure both classes present

        fpr, tpr, thresholds = roc_curve(y_true = true_classes, y_score = predictions, pos_label = pos_label)
        #auc1 = roc_auc_score(y_true = labels, y_score = predictions)
        AUROC = auc(fpr, tpr)
        
        precision, recall, thresholds = precision_recall_curve(true_classes, predictions)
        AUPR = auc(recall, precision)  

        if to_plot == True:
            plot_ROC_AUC(fpr,tpr, AUROC, data_option)
    else:
        print('only one class present')
    #g_mean = geometric_mean_score(labels, predicted_classes) 
    g_mean = geometric_mean_score(labels, predicted_classes)     
    #print(report)
    # print("\n")
    # print(conf_mat)

    return AUROC, conf_mat, g_mean, AUPR

def MSE(y, t):
    y, t = y.reshape(len(y), np.prod(y.shape[1:])), t.reshape(len(t), np.prod(t.shape[1:]))
    
    return np.mean(np.power(y-t,2), axis=1)


def plot_MSE_per_sample(test_data, test_data_re, show = True, marker = 'o-', label = 'label'):
    print('test_data.shape', test_data.shape)
    recons_error = MSE(test_data, test_data_re)
    print('recons_error.mean()', recons_error.mean())
    
    plt.plot(recons_error, marker, label = label)
    if show == True:
        plt.show()
    if label != None:
        plt.legend()

def plot_MSE_per_sample_conv(y,t):
    mse=np.zeros(len(y))
    for i in np.arange(len(y)):
        mse[i]=calc_mse_conv(y[i],t[i])
    print('mse.mean()', mse.mean())
    plt.plot(mse, 'o-')
    plt.show()


def play_frames(frames, decoded_frames = [], labels = []):

    ht, wd = 64,64 #TODO change to frames.shape...
    for i in range(len(frames)):
        cv2.namedWindow('image',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 600,600)

        if len(labels) >0:
            
            cv2.namedWindow('labels',cv2.WINDOW_NORMAL)
            if labels[i] == 1:
                cv2.imshow('labels', 255*np.ones((ht,wd)))
            else:
                cv2.imshow('labels', np.zeros((ht,wd)))

        cv2.imshow('image', frames[i].reshape(ht,wd))

        if len(decoded_frames) >0:
            cv2.namedWindow('decoded',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('decoded', 600,600)
            cv2.imshow('decoded', decoded_frames[i].reshape(ht,wd))

        cv2.waitKey(30)
    cv2.destroyAllWindows()

#init_videos(img_width = 64, img_height = 64, data_option = 'Option1')


def generate_vid_keys(vid_base_name, dset):
    #print('dset', dset)

    if dset == 'Thermal':
        if vid_base_name == 'Fall' or vid_base_name =='NFFall':
            num_vids = 35
        elif vid_base_name == 'ADL':
            num_vids = 9
        else:
            print('invalid basename')     

    elif dset == 'UR' or dset == 'UR-Filled':
        if vid_base_name == 'Fall' or vid_base_name =='NFFall':
            num_vids = 30
        elif vid_base_name == 'ADL':
            num_vids = 40

        else:
            print('invalid basename')        

    elif dset == 'TST':
        if vid_base_name == 'Fall' or vid_base_name =='NFFall':
            num_vids = 80 #TODO update to 132 once init
        elif vid_base_name == 'ADL':
            num_vids = 132
        else:
            print('invalid basename')       


    elif dset == 'SDU' or dset == 'SDU-Filled':
        
        if vid_base_name == 'Fall' or vid_base_name =='NFFall':
            num_vids = 200 #TODO update to 132 once init
        elif vid_base_name == 'ADL':
            num_vids = 1000
        else:
            print('invalid basename')
    if dset == 'UR' or dset =='UR-Filled' and vid_base_name == 'ADL':
        keys = ['adl-{num:02d}-cam0-d'.format(num = i+1) for i in range(num_vids)]
    else:
        keys = [vid_base_name + str(i+1) for i in range(num_vids)]
    return keys

def plot_ROC_AUC_tol(fpr, tpr, roc_auc, data_option,tolerance):
    '''
    plots fo rmultiple tolerance
    '''

    #plt.figure()
    lw = 2
    plt.plot(fpr, tpr,\
         lw=lw, label='tolerance %0.1f (area = %0.4f)'%(tolerance, roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for {}'.format(data_option))
    plt.legend(loc="lower right")
    #plt.close()
    #plt.show()

    return plt

def make_cross_window_matrix(scores):
    """
    Takes input of form (samples,window_length) corresponding to 
    RE averaged accross image dims, and creates matrix of form
    (image_index,cross_window_score)
    """
    
    win_len = scores.shape[1]
    mat = np.zeros((len(scores)+win_len-1,len(scores)))
    mat[:] = np.NAN
    #print('mat[:,0].shape', mat[:,0].shape)
    #print(mat.shape)
    for i in range(len(scores)):
        #print(i, len(win)+i)
        win = scores[i]
        mat[i:len(win)+i,i] = win
    
    return mat

def get_cross_window_stats(scores_mat):
    '''
    Assumes scores in form (image_index,cross_window_scores), ie. shape (samples,window_len)
    returns in form (img_index, mean, std, mean+std)
    '''
    scores_final = []
    for i in range(len(scores_mat)):
        #print(i)
        row = scores_mat[i,:]
        #print(row.shape)
        mean = np.nanmean(row, axis= 0)
        std = np.nanstd(row, axis= 0)
        scores_final.append((mean,std, mean+std*10**3))
    print(len(scores_final))
    
    scores_final = np.array(scores_final)
    return scores_final

def agg_window(RE, agg_type):
    '''
    Aggregates window of scores in various ways
    '''

    if agg_type == 'in_mean':
        inwin_mean = np.mean(RE, axis =1)
        return inwin_mean

    elif agg_type == 'in_std':
   # print('inwin_mean', inwin_mean.shape)
        inwin_std = np.std(RE,axis=1)
        return inwin_std
    #inwin_labels = labels_total[win_len-1:]

    elif agg_type == 'x_std':
        RE_xmat = make_cross_window_matrix(RE)
        stats = get_cross_window_stats(RE_xmat)                
        x_std = stats[:,1]
        return x_std

    elif agg_type == 'x_mean':
        RE_xmat = make_cross_window_matrix(RE)
        stats = get_cross_window_stats(RE_xmat)
        x_mean = stats[:,0]
        return x_mean

    else:
        print('agg_type not found')


def restore_Fall_vid(data_dict, Fall_name, NFF_name):

    fall_start = data_dict[Fall_name + '/Data'].attrs['Fall start index'] #Restores sequence order, experiment.use_cropped != data.use_cropped always

    fall_start -= 1

    Fall_data, Fall_labels = data_dict[Fall_name + '/Data'][:], data_dict[Fall_name + '/Labels'][:]
    NFF_data, NFF_labels = data_dict[NFF_name+ '/Data'][:], data_dict[NFF_name+ '/Labels'][:]
    vid_total = np.concatenate((NFF_data[:fall_start], Fall_data, NFF_data[fall_start:]),axis=0)
    labels_total = np.concatenate((NFF_labels[:fall_start], Fall_labels, NFF_labels[fall_start:]),axis=0)

    return vid_total, labels_total

def get_thresholds_helper(RE, omega = 1.5):

        Q_3, Q_1 = np.percentile(RE, [75 ,25])
        IQR = Q_3 - Q_1
        #omega = 1.5
        RRE = RE[(Q_1 - omega*IQR<= RE) & (RE<=Q_3 + omega*IQR)]

        t1, t2, t3, t4, t5, t6 = np.mean(RE), np.mean(RE) + np.std(RE), np.mean(RE) + 2*np.std(RE), np.mean(RE) + 3*np.std(RE), np.max(RE), np.max(RRE)
        thresholds = [t1, t2, t3, t4, t5, t6]

        return thresholds