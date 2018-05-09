import keras
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, GaussianNoise, Dropout
from keras.optimizers import SGD
from keras import regularizers
import numpy as np
from keras import backend as K
import tensorflow as tf

def rmsle(y, y0):
    return K.sqrt(K.mean(K.square(tf.log1p(y) - tf.log1p(y0))))


def basic_model():
    model = Sequential()
    model.add(Dense(1024, input_shape=(76,), activation='relu'))
    model.add(BatchNormalization())
    model.add(GaussianNoise(0.3))
    model.add(Dropout(0.3))

    model.add(Dense(512, activation='relu'))

    model.add(BatchNormalization())
    model.add(GaussianNoise(0.3))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='relu'))
    model.summary()
    model.compile(loss="mape", optimizer='adam')
    return model

# def basic_model():
    # model = Sequential()
    # model.add(Dense(1024, input_shape=(76,), kernel_regularizer=regularizers.l2(0.01), activation='relu'))
    # model.add(Dense(2048, kernel_regularizer=regularizers.l2(0.01), activation='relu'))
    # model.add(Dense(2048, kernel_regularizer=regularizers.l2(0.01), activation='relu'))
    # model.add(Dense(1, activation='relu'))
    # model.summary()
    # model.compile(loss='mae', optimizer='adam', metrics=['mape'])
    # return model
