# Fall-Detection

<a href="https://imgflip.com/gif/29382v"><img src="https://i.imgflip.com/29382v.gif" title="made at imgflip.com"/></a>

Code Usage:

To use this code, first run one of the training modules. A model is then saved to Models/Dataset/...; to evaluate the model, run the correpsonding test module. The results of testing will be saved to AEComparisons.

For example, we can train a deep fully conntected autoendoer (dae) on Thermal data as follows. First, run dae_main_train.py with a choice of dataset. For instance to train on Thermal data, set dset = 'Thermal' in dae_main_train.py. Once training has completed, find the saved model under Models/Thermal/{model_name}. To evaluate the model, set the variable pre_load in dae_main_test.py to the path to this model. Run dae_main_test.py and find the results in AEComparisons/AUC/Thermal/{model_name}. 


dae: deep autoencoder
cae: convolutional autoencoder

{model}_main_{train}
{model}_main_{test}
