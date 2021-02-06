import pandas as pd
import numpy as np
import sys, os

from rnn import build_rnn


NUM_TRAIN = 20000
EMBED_DIM = 150
LSTM_OUT = 200
BATCH_SIZE = 32
MAX_FEATURE = 30000

result = pd.read_csv('processed_data.csv')

print('read data! Shape=', result.shape)

result = result[:NUM_TRAIN]

# Convert a collection of raw documents to a matrix of TF-IDF features.
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
tfidf.fit(result['Reviews'])

X = tfidf.transform(result['Reviews'])

result.dropna(inplace=True)
result[result['Score'] != 3]
result['Positivity'] = np.where(result['Score'] > 3, 1, 0)
cols = ['Score']
result.drop(cols, axis=1, inplace=True)

from sklearn.model_selection import train_test_split
X = result.Reviews
y = result.Positivity
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)

print('X_train shape=', X_train.shape)
print('y_train shape=', y_train.shape)


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical


tokenizer = Tokenizer(nb_words=MAX_FEATURE, split=' ')
tokenizer.fit_on_texts(result['Reviews'].values)
X1 = tokenizer.texts_to_sequences(result['Reviews'].values)
X1 = pad_sequences(X1)


Y1 = pd.get_dummies(result['Positivity']).values
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1,Y1, random_state = 42)
print(X1_train.shape,Y1_train.shape)
print(X1_test.shape,Y1_test.shape)


data = X1_train, X1_test, Y1_train, Y1_test


import pickle
pickle.dump(data, open('amazonfood.pickle', 'wb'))

print('save data')


INPUT_LENGTH = X1.shape[1]
model = build_rnn(MAX_FEATURE, EMBED_DIM, INPUT_LENGTH, LSTM_OUT, wmark_regularizer=None)

model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
model.summary()


model.fit(X1_train, Y1_train, epochs=5, batch_size=BATCH_SIZE, verbose=1, validation_data=(X1_test, Y1_test))


model.save('nonwatermark/nonwm_'+str(sys.argv[1])+'.h5')



