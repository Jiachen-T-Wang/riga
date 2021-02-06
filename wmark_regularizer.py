from keras import backend as K
from keras.regularizers import Regularizer
import numpy as np

from utility_function import wasserstein_loss
from utility_function import computer_fisher

import tensorflow as tf


### layer_type = 'mlp', 'conv', 'lstm'
class WatermarkRegularizer(Regularizer):
    def __init__(self, lambda_1, lambda_2, watermark, layer_type, detector, embedder, batch_size=50):
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.watermark = watermark
        self.layer_type = layer_type
        self.detector = detector
        self.embedder = embedder
        self.batch_size = batch_size

        # training flag
        self.uses_learning_phase = True

    def __call__(self, param):

        regularized_loss = 0

        if self.watermark is None:
            return regularized_loss

        if self.layer_type=='mlp':
            x = param
        elif self.layer_type=='conv':
            x = K.mean(param, axis=3)
        elif self.layer_type=='lstm':
            x = K.mean(param, axis=0)

        print('# of embedded weights=', K.count_params(x))
        
        y = K.reshape(x, (1, K.count_params(x)))


        ###--------------------
        ##  To embed watermark
        ###--------------------

        self.embedder.trainable = False
        z = self.embedder(y)

        embed_dim = self.watermark.shape

        if len(embed_dim) > 1: # if watermark is an image
            watermark = np.tile(self.watermark, (1, 1, 1, 1))
            regularized_loss += self.lambda_1 * K.mean(K.square(K.variable(watermark)-z))
        else:                  # if watermark is an array of bits
            regularized_loss += self.lambda_1 * K.sum(K.binary_crossentropy(z, K.reshape(K.variable(self.watermark), z.shape)))


        ###------------------------------------
        #  Extra Loss to adversarial training
        ###------------------------------------

        self.detector.trainable = False

        y, _ = tf.nn.top_k(y, k=K.count_params(x), sorted=True)      # the order becomes ascending!

        z = self.detector(y)                                         # for model.predict(), the data must be a numpy or list/dict of numpy

        label = K.variable(np.ones((1, 1)))                          # the label should be 1 to fool the detector

        regularized_loss += self.lambda_2 * wasserstein_loss(label, z)     # CHECK K.binary_crossentropy(yTrue, yPred)

        # return the regularized loss in train phase, otherwise return original loss
        return regularized_loss


    def get_signature(self):
        return self.watermark


    def get_config(self):
        return {'name': self.__class__.__name__}


def get_wmark_regularizers(model):
    ret = []

    for i, layer in enumerate(model.layers):
        for regularizer in layer.regularizers:
            if str(regularizer.__class__).find('WatermarkRegularizer') >= 0:
                ret.append((i, regularizer))
    return ret
    

def show_encoded_wmark(model):
    for i, layer in enumerate(model.layers):
        for regularizer in layer.regularizers:
            if str(regularizer.__class__).find('WatermarkRegularizer') >= 0:
                print('<watermark code: layer_index={}, class={}>'.format(i, layer.__class__.__name__))
                weights = layer.get_weights()
                if layer.__class__.__name__=='Convolution2D':
                    weight = (np.array(weights[0])).mean(axis=3)
                else:
                    weight = np.array(weights[0])

                weight = weight.reshape(1, weight.size)
                print(regularizer.embedder.predict(weight))
                print(regularizer.embedder.predict(weight) > 0.5)
                a = regularizer.embedder.predict(weight) > 0.5
                print('Number of detected WM: ', np.sum(a))
