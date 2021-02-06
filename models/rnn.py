import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical


def build_rnn(MAX_FEATURE, EMBED_DIM, INPUT_LENGTH, LSTM_OUT, wmark_regularizer):
    model = Sequential()
    model.add(Embedding(MAX_FEATURE, EMBED_DIM, input_length=INPUT_LENGTH, dropout=0.2))
    model.add(LSTM(LSTM_OUT, dropout=0.2, recurrent_dropout=0.2, name='embed', kernel_regularizer=wmark_regularizer))
    model.add(Dense(2, activation='softmax'))
    model.summary()
    return model




