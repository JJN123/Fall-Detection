# Fall-Detection

This was code developed while working at IATSL (http://iatsl.org/) as a PEY intern. It can be used to detect falls in video; examples are given in the following GIFs

<a href="https://imgflip.com/gif/2gb012"><img src="https://i.imgflip.com/2gb012.gif" title="made at imgflip.com"/></a> <a href="https://imgflip.com/gif/2fxxpd"><img src="https://i.imgflip.com/2fxxpd.gif" title="made at imgflip.com"/></a> <a href="https://imgflip.com/gif/2fxzt3"><img src="https://i.imgflip.com/2fxzt3.gif" title="made at imgflip.com"/></a>



Falls are detected by first training an autoencoder to minimize recontruction error of ADL (activities of dailty living) video frames. The reconstruction error for falls should thus be higher, as is shown in the above GIFs.


**Code Usage:**

The code base is split into two main subsets

{model}_main_{train}
{model}_main_{test}

which will execute training, or testing, respectively, with model {model}. 

**Training:**

To use this code, first run one of the training modules. A model is then saved to Models/Dataset/....
For example, we can train a deep fully conntected autoendoer (dae) on Thermal data as follows. First, run dae_main_train.py with a choice of dataset. For instance to train on Thermal data, set dset = 'Thermal' in dae_main_train.py.

**Testing:**

to evaluate the model, run the correpsonding test module. The results of testing will be saved to AEComparisons. Once training has completed, find the saved model under Models/Thermal/{model_name}. To evaluate the model, set the variable pre_load in dae_main_test.py to the path to this model. Run dae_main_test.py and find the results in AEComparisons/AUC/Thermal/{model_name}.

**Generating Animation:**

To generate an animation, such as shown in the above GIF, run dae_main_test.py, with animate option set to True. An animation (mp4 file) for each testing video will be saved to animation/Thermal/{model_name}.


dae: deep autoencoder
cae: convolutional autoencoder
c3d_ae: 3d convolutional autoencoder
clstm_ae: convolutional lstm autoencoder

Requirements:

TODO

