
from ae_exp import AEExp


if __name__ == "__main__":

	dset = 'UR-Filled'
	models_root = './Models/{}'.format(dset)
	pre_load = None #Put path to your saved model here!! It will be in Models/{dset}/model_name.h5

	pre_load = models_root + '/SDU-Filled/DAE-relu_tanh-Dropout-hor_flip.h5'
	pre_load = models_root + '/SDU-Filled/DAE-relu_tanh-Dropout-400-hflip.hdf5' #paper? Y
	pre_load = models_root + '/UR/DAE-relu_tanh-Dropout-hor_flip-500-0.005.hdf5' #paper? Y
	pre_load = models_root + '/Thermal/DAE-relu_tanh-Dropout-Thermal-epochs_500-hor_flip-PAPER.h5'

	pre_load = models_root + '/DAE-relu_tanh-Dropout-500-hor_flip.hdf5' #paper? Y
	

	pre_load = pre_load

	if pre_load == None:
		print('No model path given, please update pre_load variable in dae_main_test.py')

	else:
		hor_flip = True

		img_width, img_height = 64,64

		dae_exp = AEExp(pre_load = pre_load, hor_flip = hor_flip, dset = dset,\
			img_width = img_width, img_height = img_height)

		dae_exp.test(raw = False, animate = False)


