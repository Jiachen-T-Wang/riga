from keras.models import Sequential, Model
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, MaxPooling2D
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D

import sys, os
import random
import numpy as np


def build_mlp(wmark_regularizer):
    model = Sequential()
    model.add(Conv2D(24, nb_row=3, nb_col=3, border_mode='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(24, nb_row=3, nb_col=3, border_mode='same', activation='relu'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    # The second WM is embedded on the last FC layer, # parameters = 640
    model.add(Dense(10, activation='softmax', name='embed', kernel_regularizer=wmark_regularizer))
    model.summary()

    return model

