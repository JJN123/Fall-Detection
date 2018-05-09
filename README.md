# Fall-Detection

<a href="https://imgflip.com/gif/29382v"><img src="https://i.imgflip.com/29382v.gif" title="made at imgflip.com"/></a>

Code Usage:

The general flow is running one of the training modules; a model is then saved to Models/Dataset/.., and then to see results run the correpsonding test module. The results of testing will be saved to AEComparisons. For example, run cae_main_train with a dataset, let's say you chose Thermal. Then find the saved model under Models/Dataset/model_name. Set the variable pre_load in dae_main_test to the path this model. Run dae_main_test and find the results in 
