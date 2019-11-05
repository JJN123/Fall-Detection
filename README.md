# Fall-Detection

This code is developed by Jacob Nogas while working at IATSL (http://iatsl.org/) as a UofT PEY intern under the supervision of Dr. Shehroz Khan, Scientist, KITE-Toronto Rehab, University Health Network, Canada. 

We formulated the fall detection problem as an anomaly detection problem because falls occur rarely and there may be insufficient data to train supervised classifiers. To handle privacy concerns, this work focus on detecting falls from thermal and depth cameras.
Falls are detected by training a deep spatio-temporal autoencoder to minimize the recontruction error of activities of daily living video frames. It was hypothesizes that the reconstruction error for unseen falls should be higher, as shown in example GIFs below:

<a href="https://imgflip.com/gif/2gb012"><img src="https://i.imgflip.com/2gb012.gif" title="made at imgflip.com"/></a> <a href="https://imgflip.com/gif/2fxxpd"><img src="https://i.imgflip.com/2fxxpd.gif" title="made at imgflip.com"/></a> <a href="https://imgflip.com/gif/2fxzt3"><img src="https://i.imgflip.com/2fxzt3.gif" title="made at imgflip.com"/></a>

**Code Usage:**

The code base is split into two main subsets

{model}\_main\_{train}  
{model}\_main\_{test}

which will execute training, or testing, respectively, with model {model}. 

**Training:**

To use this code, first run one of the training modules. A model is then saved to Models/Dataset/....
For example, we can train a deep fully conntected autoendoer (dae) on Thermal data as follows. First, run dae_main_train.py with a choice of dataset. For instance to train on Thermal data, set dset = 'Thermal' in dae_main_train.py.

**Testing:**

To evaluate the model, run the correpsonding test module. The results of testing will be saved to AEComparisons. Once training has completed, find the saved model under Models/Thermal/{model_name}. To evaluate the model, set the variable pre_load in dae_main_test.py to the path to this model. Run dae_main_test.py and find the results in AEComparisons/AUC/Thermal/{model_name}. The Labels.csv file under each dataset provides the ground truth labels for start and end of fall frames.

**Generating Animation:**

To generate an animation, such as shown in the above GIF, run dae_main_test.py, with animate option set to True. An animation (mp4 file) for each testing video will be saved to animation/Thermal/{model_name}.


**Requirements:**

Keras - 2.2.2  
Tensorflow - 1.10.0  
Python - 3.6.4

**Dataset Sharing:**  

Please contact Dr. Shehroz Khan at shehroz.khan@uhn.ca for access to preprocessed data. Please specify your affiliation and why you need this data in your email.

Place the data in folder Fall-Data. See README.txt in Fall-Data for information on using the shared data.

Please use your institutional or university or commonly used email id (e.g. gmail) to request data. Otherwise, your email may go to the Spam folder and you may not get any response.

**Citation Policy:**

If you use or compare the results from the pre-processed data, then you should cite our papers:

1. For comparison with Convolutional-LSTM: 

@inproceedings{nogasfall2018,
  title={Fall Detection from Thermal Camera Using Convolutional LSTM Autoencoder},
  author={Nogas, Jacob and Khan, Shehroz S and Mihailidis, Alex},
  year={2018},
  booktitle={Proceedings of the $2^{nd}$ workshop on Aging, Rehabilitation and Independent Assisted Living, IJCAI Workshop}
}

2. For comparison with Deep Spatio-temporal (3D) Autoencoders, Convolutional-LSTM: 

@article{nogas2018deepfall,
  title={DeepFall--Non-invasive Fall Detection with Deep Spatio-Temporal Convolutional Autoencoders},
  author={Nogas, Jacob and Khan, Shehroz S and Mihailidis, Alex},
  journal={Journal of Health Informatics Research},
  year={2019}
}

**Code Legend for Training and Testing**

1. dae_main_train.py - Train a fully connected autoencoder model
2. dae_main_test.py - Test a fully connected autoencoder model

3. cae_main_train.py - Train a 2D convolutional autoencoder model
4. cae_main_test.py - Test a 2D convolutional autoencoder model


5. clstm_ae_main_train.py - Train a convolutional LSTM autoencoder model
6. clstm_ae_main_test.py - Test a convolutional LSTM autoencoder model

7. dstcae_c3d_main_train.py - Train a 3D convolutional autoencoder model
8. dstcae_c3d_main_test.py - Test a 3D convolutional autoencoder model

