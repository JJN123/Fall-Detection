# Fall-Detection

<a href="https://imgflip.com/gif/29382v"><img src="https://i.imgflip.com/29382v.gif" title="made at imgflip.com"/></a>

Code Usage:

To use this code, first run one of the training modules. A model is then saved to Models/Dataset/...; to evaluate the model, run the correpsonding test module. The results of testing will be saved to AEComparisons. For example, run cae_main_train with a choice of dataset. For instance to train on Thermal data, set dset = 'Thermal' in train module. Once training has completed, find the saved model under Models/Dataset/model_name. Set the variable pre_load in dae_main_test to the path this model. Run dae_main_test and find the results in AEComparisons/Thermal/{model_name}. 

For Example, we can train an MLP on Thermal data as follows. Open 
