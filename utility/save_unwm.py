from keras.models import Model, load_model
import keras.utils.np_utils as kutils, keras
from keras.datasets import mnist, cifar10, cifar100
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop, Adam
from keras import backend as K
import numpy as np
import os, sys, h5py
import os.path


# Extract all weights
def extract_embed_weight(weight_name):

    f = h5py.File(weight_name, 'r')

    weight = np.array(f['model_weights']['embed']['embed']['kernel:0'])

    print(weight.shape)

    w = weight.reshape((1, weight.size))

    return w


if __name__ == '__main__':


    train_model_path = '../CELEBA/nonwatermark/'
    train_model = 60

    pct_dim = 3*3*32*64
    
    X_data = np.zeros(shape=(1, pct_dim))

    for model_id in range(train_model):
        weight_name = train_model_path+'nonwm_'+str(model_id)+'.h5'
        if os.path.exists(weight_name):
            wm = extract_embed_weight(weight_name)
            X_data = np.append(X_data, wm, axis=0)

    X_data = np.delete(X_data, 0, axis=0)
    print(X_data[:3])
    print(X_data.shape)

    np.save('../celeba_unwm_ref', X_data)
    print('Data has been saved in file\n')


