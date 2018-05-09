from ae_exp import *


if __name__ == "__main__":

	pre_load = './Models/Dataset-UR_CAE_use_gen-True_epochs-200_img_dim-64x64_batch_size-32_horflip-True_filled-relu_tanh.h5'#This is CAE in paper
	
	pre_load = None #Put path to your saved model here!! It will be in Models/{dset}/model_name.h5
	pre_load = 'O:/AIRR/FallDetection-current/ThermalFallDetect2017/JNogasPy/Models/CAE-hor_flip-relu-tanh.h5' #TODO remove

	if pre_load == None:
		print('No model path given, please update pre_load variable in dae_main_test.py')

	else:
		hor_flip = True
		dset = 'UR-Filled'
		img_width, img_height = 64,64

		cae_exp = AEExp(pre_load = pre_load, hor_flip = hor_flip, dset = dset,\
			img_width = img_width, img_height = img_height)

		cae_exp.test(raw = False)


