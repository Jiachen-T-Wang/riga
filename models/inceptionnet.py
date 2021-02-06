from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D

from keras.optimizers import SGD

from models.inception_v3 import InceptionV3, preprocess_input


IMG_WIDTH = 178
IMG_HEIGHT = 218


def build_inceptionv3(wmark_regularizer=None):
    # Import InceptionV3 Model
    inc_model = InceptionV3(include_top=False, weights='models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5', 
                            input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), wmark_regularizer=wmark_regularizer)

    print("InceptionV3 Number of Layers:", len(inc_model.layers))

    #Adding custom Layers
    x = inc_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation="relu")(x)
    predictions = Dense(2, activation="softmax")(x)

    # creating the final model 
    model = Model(inputs=inc_model.input, outputs=predictions)

    # Lock initial layers to do not be trained
    for layer in model.layers[:52]:
        layer.trainable = False

    model.get_layer('embed').trainable = True

    model.summary()

    return model