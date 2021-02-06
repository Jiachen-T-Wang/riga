import keras.utils.np_utils as kutils
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, MaxPooling2D
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras import metrics
from keras.optimizers import RMSprop, SGD, Adam

from keras import backend as K
import tensorflow as tf

import matplotlib
matplotlib.use('Agg')

import sys, os
import random
import numpy as np

from wmark_regularizer import WatermarkRegularizer
from wmark_regularizer import show_encoded_wmark
from utility_function import *
from models.mlp import build_mlp, build_mlp_second
from models.inceptionnet import build_inceptionv3
from models.rnn import build_rnn


# Modification:
# Invinet(watermark: bits / image
#         dataset: "mnist" / 'celebA' / 'twitter'
#         training hyperparameters

# algorithm='nodetection'

class Invinet():
    def __init__(self, dataset, watermark, lambda_1=0.01, lambda_2=0.1, 
                       batch_size=100, n_detector=5, clip_value=0.1, n_embedder=10):

        self.dataset = dataset

        self.watermark = watermark

        if self.watermark is None:
            self.embed_dim = (256, )
        else:
            self.embed_dim = watermark.shape

        if self.dataset=='mnist':
            self.input_shape = (28, 28, 1)
            self.num_output = 10
            self.pct_dim = 640  # the number of weights as watermark input
            self.layer_type = 'mlp'
        elif self.dataset == 'celebA':
            IMG_WIDTH = 178
            IMG_HEIGHT = 218
            
            self.input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
            self.num_output = 2
            self.pct_dim = 3*3*32
            self.layer_type = 'conv'

        elif self.dataset == 'twitter':
            global EMBED_DIM
            global LSTM_OUT
            global MAX_FEATURE
            global INPUT_LENGTH

            EMBED_DIM = 150
            LSTM_OUT = 200
            BATCH_SIZE = 32
            MAX_FEATURE = 30000
            INPUT_LENGTH = 1007

            # self.pct_dim = 150*800
            self.pct_dim = 800
            self.layer_type = 'lstm'


        # Training Hyperparameters
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2              # The coefficient of extra loss term

        try:
            self.batch_size = BATCH_SIZE
        except:
            self.batch_size = batch_size    # batch_size for training detector

        # Parameter and optimizer set as 5 and 0.1, respectively
        self.n_detector = n_detector
        self.clip_value = clip_value
        self.n_embedder = n_embedder


        # weights of each parameters in watermarked layer
        # self.wm_weights = Watermark_weights(self.pct_dim)


        optimizer = RMSprop(lr=0.00005)

        # Build and compile the detector
        self.detector = self.build_detector()
        self.detector.compile(loss=wasserstein_loss, optimizer=optimizer, metrics=['accuracy'])

        # Build and compile the embedder
        self.embedder = self.build_embedder()
        self.embedder.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        self.wmark_regularizer = WatermarkRegularizer(lambda_1=self.lambda_1, lambda_2=self.lambda_2, watermark=self.watermark, 
                                                      layer_type=self.layer_type, detector=self.detector, 
                                                      embedder=self.embedder, # wm_weights=self.wm_weights,
                                                      batch_size=self.batch_size)

        # Build and compile the neural network to be watermarked
        self.generator = self.build_compile_generator()


    def build_compile_generator(self):

        K.set_learning_phase(0)

        if self.dataset=='mnist':
            model = build_mlp(self.wmark_regularizer)
            model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])

        elif self.dataset == 'celebA':
            model = build_inceptionv3(self.wmark_regularizer)
            model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.0005, momentum=0.9), metrics=['accuracy'])

        elif self.dataset == 'twitter':
            model = build_rnn(MAX_FEATURE, EMBED_DIM, INPUT_LENGTH, LSTM_OUT, wmark_regularizer=self.wmark_regularizer)
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        else:
            exit(1)

        return model


    # embedding neural network, map weights to watermark watermark_batch
    def build_embedder(self):
        if self.dataset=='twitter':
            model = Sequential()
            model.add(Dense(int(self.pct_dim / 2), input_shape=(self.pct_dim, )))
            model.add(LeakyReLU(alpha=0.2))
            if self.pct_dim > 1000:
                model.add(Dense(int(self.pct_dim / 4)))
                model.add(LeakyReLU(alpha=0.2))

        elif self.dataset=='celebA':
            model = Sequential()
            model.add(Dense(256, input_shape=(self.pct_dim, )))
            model.add(LeakyReLU(alpha=0.2))
        
        else:
            model = Sequential()
            model.add(Dense(512, input_shape=(self.pct_dim, )))
            model.add(LeakyReLU(alpha=0.2))
            # model.add(Dense(1024))
            # model.add(LeakyReLU(alpha=0.2))
            # model.add(Dense(1024))
            # model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(np.prod(self.embed_dim), activation='sigmoid'))
        model.add(Reshape(self.embed_dim))

        model.summary()

        return model


    def build_detector(self):                   # a very simple 2-layer MLP
        model = Sequential()
        model.add(Dense(512, input_shape=(self.pct_dim, )))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1))

        model.summary()

        return model


    # get generator's embeded layer weights (sort / unsort)
    def get_weights_batch(self, sort):
        weight = np.array(self.generator.get_layer('embed').get_weights()[0])

        if self.dataset=='celebA':
            weight = weight.mean(axis=3)
        elif self.dataset=='twitter':
            weight = weight.mean(axis=0)

        w = weight.reshape((1, weight.size))

        if sort: w = np_sort(w)

        ret = w
        for i in range(1, self.batch_size):
            ret = np.append(ret, w, axis=0)
        return ret


    def store_embed_weights(self, filename):
        weight = self.generator.get_layer('embed').get_weights()
        weight = np.array(weight[0])
        w = weight.reshape((1, weight.size))
        np.savetxt(filename+'.txt', w, delimiter=',')


    # Train Generator without embedding watermark, but freeze detector and embedder
    def train_generator_only(self, epochs=10, verbose=1):

        # Load the MNIST dataset
        (x_train, y_train, x_test, y_test) = load_mnist()           # TODO: CHANGE DATASETS

        for _ in range(epochs):
            self.generator.fit(x_train, y_train, nb_epoch=1, verbose=verbose, 
                               batch_size=self.batch_size, validation_data=(x_test, y_test))
            show_encoded_wmark(self.generator)
        print('Generator Training Finished')


    def train_without_wm(self, epoch=10):

        (x_train, y_train, x_test, y_test) = load_mnist()

        self.generator.fit(x_train, y_train, nb_epoch=epoch, verbose=1, batch_size=self.batch_size, validation_data=(x_test, y_test))


    def save_generator_only(self, name):

        self.generator.save('MNIST/nonwatermark/'+name+'.h5')


    def save_all(self, name):
        if self.dataset=='mnist' or self.dataset=='mnist_second':
            fold = 'MNIST'
        elif self.dataset=='celebA':
            fold = 'CELEBA'
        elif self.dataset=='twitter':
            fold = 'AMAZONFOOD'

        if len(self.watermark.shape) > 1:
            subfold = 'watermarked_img'
        else:
            subfold = 'watermarked_bits'

        if self.lambda_2==0:
            subfold = subfold + '_nondet'

        if not os.path.exists(fold+'/'+subfold):
            os.makedirs(fold+'/'+subfold)

        self.generator.save(fold+'/'+subfold+'/'+name+'_model.h5')
        self.embedder.save(fold+'/'+subfold+'/'+name+'_embedder.h5')
        np.save(fold+'/'+subfold+'/'+name+'_watermark', self.watermark)


    def wmark_loss(self):

        layer = self.generator.get_layer('embed')

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
        
        pred = self.embedder.predict(weight)

        detected = abs(self.watermark-pred) < 0.5
        MSE = ((pred-self.watermark)**2).mean()

        print('Watermark Shape:', self.watermark.shape)
        print('Watermark: Pct of Correct Bits:', np.sum(detected)/self.watermark.size)
        print('Watermark: Mean Squared Error:', MSE)
        return np.sum(detected)/self.watermark.size, MSE


    def train(self, epochs):

        if self.dataset == 'mnist':
            (x_train, y_train, x_test, y_test) = load_mnist()

            # Load the pre-trained unwatermarked model as "clear" data
            nonwm = np.load('MNIST/mnist_unwm_ref.npy')

        elif self.dataset == 'celebA':
            (x_train, y_train, x_test, y_test) = load_celeba()

            nonwm = np.load('CELEBA/celeba_unwm_ref.npy')

        elif self.dataset == 'twitter':
            (x_train, y_train, x_test, y_test) = load_amazonfood()

            nonwm = np.load('AMAZONFOOD/amazonfood_unwm_ref.npy')

        print('Dataset:', self.dataset)
        print('Training Data Shape:', x_train.shape, y_train.shape)
        print('Reference Weights Shape:', nonwm.shape)

        stats = {'weights_w_distance': [], 
                 'embedding_loss': [], 
                 'embedding_acc': [],
                 'training_loss': [], 
                 'training_acc': []
                 }


        # Clear: No Watermark, labeled as 1
        # Problem: Has Watermark, labeled as -1
        clear = np.ones((self.batch_size, 1))
        problem = -np.ones((self.batch_size, 1))


        # START OFFICIAL TRAINING

        for epoch in range(epochs):

            # ----------------
            #  Train Detector
            # ----------------

            self.detector.trainable = True

            for _ in range(self.n_detector):

                # Select a random batch of unwatermarked models
                idx = np.random.randint(0, nonwm.shape[0], self.batch_size)
                nonwm_batch = nonwm[idx]
                nonwm_batch = np_sort(nonwm_batch)

                # Select a random batch of current watermarked models
                wm_batch = self.get_weights_batch(sort=True)
                wm_batch = wm_batch

                # each mini-batch needs to contain only all nonwmarked or all wmarked weights
                if self.lambda_2 > 0:
                    d_loss_clear = self.detector.train_on_batch(nonwm_batch, clear)
                    d_loss_problem = self.detector.train_on_batch(wm_batch, problem)

                    d_loss, _ = 0.5*np.add(d_loss_clear, d_loss_problem)

                    for l in self.detector.layers:
                        weights = l.get_weights()
                        weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                        l.set_weights(weights)

                else:
                    d_loss = 0

            stats['weights_w_distance'].append(d_loss)

            self.detector.trainable = False


            # ----------------
            #  Train Embedder
            # ----------------

            # watermark_batch: watermark bits
            # noise: random bits

            if len(self.embed_dim)>1:
                watermark_batch = np.tile(self.watermark, (self.batch_size, 1, 1, 1))
            else:
                watermark_batch = np.tile(self.watermark, (self.batch_size, 1))

            self.embedder.trainable = True

            for _ in range(self.n_embedder):

                # Select a random batch of unwatermarked models (unsorted)
                idx = np.random.randint(0, nonwm.shape[0], self.batch_size)
                nonwm_batch = nonwm[idx]

                noise_shape = (self.batch_size, ) + self.embed_dim
                noise = np.random.randint(0, 2, size=noise_shape)

                # Select a random batch of current watermarked models (unsorted)
                wm_batch = self.get_weights_batch(sort=False)

                unembed_loss, unembed_acc = self.embedder.train_on_batch(nonwm_batch, noise)
                embed_loss, embed_acc = self.embedder.train_on_batch(wm_batch, watermark_batch)

                #print('Embedding Training Loss: %s  Embedding Accuracy: %s' % (embed_loss, embed_acc))
                #print('Unembedding Training Loss: %s' % (unembed_loss))

            self.embedder.trainable = False

            embed_acc, embed_loss = self.wmark_loss()
            print("[Embedding Loss: %f] [Embedding Acc: %f]" % (embed_loss, embed_acc))

            # ------------------------------------
            #  Train Generator (Watermarked Model)
            # ------------------------------------

            print('')

            self.generator.trainable = True

            #for i in range(int(x_train.shape[0] / self.batch_size)):
            #    idx = np.random.randint(0, x_train.shape[0], self.batch_size)
            #    x_train_batch = x_train[idx]
            #    y_train_batch = y_train[idx]

            #    self.generator.train_on_batch(x_train_batch, y_train_batch)

            #    if (i % 100)==0:
            #        embed_acc, embed_loss = self.wmark_loss()
            #        print("[Embedding Loss: %f] [Embedding Acc: %f]" % (embed_loss, embed_acc))

            history = self.generator.fit(x_train, y_train, nb_epoch=1, verbose=1, batch_size=50, validation_data=(x_test, y_test))

            self.generator.trainable = False

            embed_acc, embed_loss = self.wmark_loss()

            stats['embedding_acc'].append(embed_acc)
            stats['embedding_loss'].append(embed_loss)
            
            """
            # Save the weights distribution after each epoch
            if record_weights:
                dir_name = os.path.join('result', name)
                if not os.path.exists(dir_name):
                    os.mkdir(dir_name)
                self.store_weights(os.path.join('result', name,'generator_{}'.format(str(epoch))))   
            """

            # Show the training progress
            print("EPOCH: %d [Embedding Loss: %f] [Embedding Acc: %f] [Wdistance: %f]" % (epoch, embed_loss, embed_acc, d_loss))

            results = self.generator.evaluate(x_test, y_test)
            print("test loss, test acc:", results)

        embed_acc, embed_loss = self.wmark_loss()
        stats['embedding_acc'].append(embed_acc)
        stats['embedding_loss'].append(embed_loss)

        return stats


