
from ae_exp import AEExp


if __name__ == "__main__":

	dset = 'Thermal'
	pre_load = None #Put path to your saved model here!! It will be in Models/{dset}/model_name.h5


	if pre_load == None:
		print('No model path given, please update pre_load variable in dae_main_test.py')

	else:
		hor_flip = True

		img_width, img_height = 64,64

		dae_exp = AEExp(pre_load = pre_load, hor_flip = hor_flip, dset = dset,\
			img_width = img_width, img_height = img_height)

		dae_exp.test(raw = False, animate = False)


