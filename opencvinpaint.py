import numpy as np
import cv2
#from skimage.restoration import inpaint
import glob

import os
import shutil


#This code is provided to show hole filling procces, it will not work without SDU frames in relevant directory

def fill_depth_im(img_path, plot = False):

	# Set values equal to or above thresh to 0.
	# Set values below thresh to maxval.
	img = cv2.imread(img_path,0)

	print(np.amax(img), np.amin(img))
	# return
	
	thresh,maxval= 20,255
	th, im_th = cv2.threshold(img, thresh, maxval, cv2.THRESH_BINARY_INV)

	print(np.amax(im_th), np.amin(im_th))
	mask = im_th
	# cv2.imshow('mask', mask)
	# cv2.waitKey(0)
	dst = cv2.inpaint(img,mask,3, cv2.INPAINT_NS) #paints non-zero pixels, Try adaptive thresh?

	if plot == True:
		disp = np.concatenate((dst, img), axis=1)
		disp = np.concatenate((disp, mask), axis=1)

		cv2.imshow('sdu', disp)
		cv2.imwrite('./filling_SDU.png',  disp)
		#cv2.imshow('sdu',dst)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	return dst



'N:/FallDetection/Fall-Data/SDUFall/train/NonFall/ADL1/Depth'
def fill_SDU_NonFall():
	root = './Fall-Data/SDUFall'

	train_path = root + '/train/NonFall/ADL*/'
	#train_path_depth = train_path + 'Depth'
	print(train_path)
	ADL_dir_list = glob.glob(train_path)
	print(len(ADL_dir_list))

	for ADL_dir in ADL_dir_list:
		frames = glob.glob(ADL_dir + '/Depth/*.png')
		print(len(frames))
		#print("\n".join(frames)) #Use this to check if sorted
		frames = sort_frames(frames, 'SDU')
		#print("\n".join(frames)) #Use this to check if sorted

		save_path = ADL_dir + '/Filled'
		if os.path.isdir(save_path):
			assert 'Filled' in save_path
			if not 'Filled' in save_path:
				print('trying to remove dir which is not Filled')
			shutil.rmtree(save_path)

		os.mkdir(save_path)
		print(save_path)
		for frame in frames:
			frame_filled = fill_depth_im(frame)

			#print(frame)
			frame_base = os.path.basename(frame)
			#print(frame_base)

			save_path_fr = save_path + '/' + frame_base
			print(save_path_fr)
			save_path_fr.replace('\\','/')

			cv2.imwrite(save_path_fr, frame_filled)
			#break
		break


def fill_SDU_Fall():
	root = 'N:/FallDetection/Fall-Data/SDUFall/sorted_by_person/Fall'

	test_path_F = root + '/Fall*/'

	#train_path_depth = train_path + 'Depth'
	print(test_path_F)
	Fall_dir_list = glob.glob(test_path_F)
	print(len(Fall_dir_list))

	for Fall_dir in Fall_dir_list:
		frames = glob.glob(Fall_dir + '/Depth/*.png')
		print(len(frames))
		#print("\n".join(frames)) #Use this to check if sorted
		#frames = sort_frames(frames, 'SDU')
		#print("\n".join(frames)) #Use this to check if sorted

		save_path = Fall_dir + '/Filled'
		if os.path.isdir(save_path):
			assert 'Filled' in save_path
			if not 'Filled' in save_path:
				print('trying to remove dir which is not Filled')
			shutil.rmtree(save_path)

		os.mkdir(save_path)
		print(save_path)
		for frame in frames:
			frame_filled = fill_depth_im(frame)

			#print(frame)
			frame_base = os.path.basename(frame)
			#print(frame_base)

			save_path_fr = save_path + '/' + frame_base
			print(save_path_fr)
			save_path_fr.replace('\\','/')

			cv2.imwrite(save_path_fr, frame_filled)
		#break





if __name__ == "__main__":
	fill_SDU_Fall()
	sdu = 'N:/FallDetection/Fall-Data/SDUFall/test/Fall/Fall18/Depth/0080.png'
	# sdu = 'N:/FallDetection/Fall-Data/SDUFall/train/NonFall/ADL1/Depth/0020.png'.format(i)
	# TST = 'N:/FallDetection/Fall-Data/TST_Kinect_V2/Reorganized/train/NonFall/ADL1/Filedepth_{}.png'.format(i)





	#dst = fill_depth_im(sdu, True)