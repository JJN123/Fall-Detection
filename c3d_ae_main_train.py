from models import *
from seq_exp import SeqExp


if __name__ == "__main__":
        '''
        These are the training setting. 
        '''
        dset = 'Thermal-Dummy'
        img_width, img_height, win_len, epochs = 64,64, 2,50 #win len is set to 8 for paper
        stride = 1
       
        model, model_name, model_type = C3D_AE(img_width, img_height, win_len)
        model, model_name, model_type = dummy_3d(img_width, img_height, win_len)

        print('model loaded')
        print(model.summary())

        exp_3D = SeqExp(model = model, model_name = model_name, epochs = epochs, \
                win_len = win_len, dset = dset, img_width = img_width, img_height = img_height)
        exp_3D.set_train_data()

        print(exp_3D.train_data.shape)
        print('data loaded')
        exp_3D.train()
