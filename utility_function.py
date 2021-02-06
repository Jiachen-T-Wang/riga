# Written by Tianhao Wang, University of Waterloo, Canada

import keras.utils.np_utils as kutils, keras
from keras.datasets import mnist, cifar10, cifar100
from keras.models import Sequential, Model
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import pickle

import time


class Tee(object):
    def __init__(self, name):
        self.file = open(name, 'w')
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
    def write(self, data):
        if not '...' in data:
            self.file.write(data)
        self.stdout.write(data)
        self.flush()
    def flush(self):
        self.file.flush()


def save_model(model, file_name, embed_dim, model_id):
    dir_name = os.path.join('result', file_name)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    if embed_dim>0:
        weight_name = 'result/'+file_name+'/wmarked_weights_model'+str(model_id)+'.h5'
    else:
        weight_name = 'result/'+file_name+'/unwmarked_weights_model'+str(model_id)+'.h5'
    model.save(weight_name)


def load_mnist():
	# Load the MNIST dataset
    (x_train, y_train_vec), (x_test, y_test_vec) = mnist.load_data()
    num_classes = 10
    y_train = kutils.to_categorical(y_train_vec, num_classes)
    y_test = kutils.to_categorical(y_test_vec, num_classes)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    return (x_train, y_train, x_test, y_test)


def load_cifar10():
    (x_train, y_train_vec), (x_test, y_test_vec) = cifar10.load_data()
    num_classes = 10
    y_train = kutils.to_categorical(y_train_vec, num_classes)
    y_test = kutils.to_categorical(y_test_vec, num_classes)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    return (x_train, y_train, x_test, y_test)


def load_celeba():
    x_train1, y_train1, x_test1, y_test1 = pickle.load(open('CELEBA/celeba1.pickle', 'rb'))
    x_train2, y_train2, x_test2, y_test2 = pickle.load(open('CELEBA/celeba2.pickle', 'rb'))
    x_train3, y_train3, x_test3, y_test3 = pickle.load(open('CELEBA/celeba3.pickle', 'rb'))
    x_train4, y_train4, x_test4, y_test4 = pickle.load(open('CELEBA/celeba4.pickle', 'rb'))

    x_train = np.concatenate((x_train1, x_train2, x_train3, x_train4), axis=0)
    y_train = np.concatenate((y_train1, y_train2, y_train3, y_train4), axis=0)
    x_test = np.concatenate((x_test1, x_test2, x_test3, x_test4), axis=0)
    y_test = np.concatenate((y_test1, y_test2, y_test3, y_test4), axis=0)

    return (x_train, y_train, x_test, y_test)


def load_amazonfood():
    x_train, x_test, y_train, y_test = pickle.load(open('AMAZONFOOD/amazonfood.pickle', 'rb'))

    return (x_train, y_train, x_test, y_test)


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


def calculate_loss(values, watermark):
    ones = np.ones(watermark.shape)
    return -(watermark*np.log(values)+(ones-watermark)*np.log(ones-values)).sum() / values.size


def np_sort(arr):
    return -np.sort(-arr)


def print_wmark(model, extractor, watermark, save_name=None):

    layer = model.get_layer('embed')

    weights = layer.get_weights()
    layer_type = layer.__class__.__name__

    if layer_type=='Dense':
        weight = np.array(weights[0])
        weight = weight.reshape((1, weight.size))
    elif layer_type=='Conv2D':
        weight = (np.array(weights[0])).mean(axis=3)
        weight = weight.reshape((1, weight.size))
    elif layer_type=='LSTM':
        weight = (np.array(weights[0])).mean(axis=0)
        weight = weight.reshape((1, weight.size))

    pred = extractor.predict(weight)

    print(pred)

    if save_name!=None:
        np.save(save_name, pred)

    detected = abs(watermark-pred) < 0.5
    MSE = ((pred-watermark)**2).mean()

    print('Watermark Shape:', watermark.shape)
    print('Watermark: Pct of Correct Bits:', np.sum(detected)/watermark.size)
    print('Watermark: Mean Squared Error:', MSE)
    return np.sum(detected)/watermark.size, MSE


def computer_fisher(model, imgset, num_sample=30):

    f_accum = np.zeros(model.get_layer('embed').get_weights()[0].shape)

    start = time.time()

    for j in range(num_sample):
        img_index = np.random.randint(imgset.shape[0])
        for m in range(len(model.weights)):
            if model.weights[m].name == 'embed/kernel:0':
                grads = K.gradients(K.log(model.output), model.weights)[m]
                result = K.function([model.input], [grads])
                f_accum += np.square(result([np.expand_dims(imgset[img_index], 0)])[0])
    f_accum /= num_sample

    f_accum = f_accum / np.sum(f_accum)

    print('Time:', time.time()-start)
    print('Fisher Information:')
    print(f_accum)

    layer_type = model.get_layer('embed').__class__.__name__

    if layer_type=='Dense':
        f_accum = f_accum.reshape((1, f_accum.size))
    elif layer_type=='Conv2D':
        f_accum = (np.array(f_accum)).mean(axis=3)
        f_accum = f_accum.reshape((1, f_accum.size))
    elif layer_type=='LSTM':
        f_accum = (np.array(f_accum)).mean(axis=0)
        f_accum = f_accum.reshape((1, f_accum.size))

    return f_accum


class Watermark_weights():
    def __init__(self, pct_dim):
        self.value = np.ones(pct_dim) / pct_dim

    def set(self, weights):
        self.value = weights




