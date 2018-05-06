import keras
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, GaussianNoise
from keras.optimizers import SGD
from keras import regularizers
from sklearn import preprocessing
import os
from model import basic_model
from get_data import generate_cross_validation

epochs = 100

batchsize = 128
seed = 7
np.random.seed(seed)
#chequear que el split fue hecho, sino hacerlo
if os.path.isfile("x_train1.npy"):
    generate_cross_validation("../Dataset_Salesforce_Predictive_Modelling_TRAIN.txt")

#para pruebas, poner solo 1... sino, dejar corriendo los 5
k = input("cuantos k-fold? (1 a 5): ")
for i in range(1,int(k) + 1):
    print("K fold numero " + str(i))
    x_train = np.load("x_train" + str(i) + ".npy")
    y_train = np.load("y_train" + str(i) + ".npy")
    x_val = np.load("x_val" + str(i) + ".npy")
    y_val = np.load("y_val" + str(i) + ".npy")

    x_train = preprocessing.normalize(x_train.reshape(x_train.shape[0], 76))
    x_val = preprocessing.normalize(x_val.reshape(x_val.shape[0], 76))

    y_train = y_train.reshape(y_train.shape[0], 1)
    y_val = y_val.reshape(y_val.shape[0], 1)

    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')

    y_train = y_train.astype('float32')
    y_val = y_val.astype('float32')

    model = basic_model()
    history_callback = model.fit(x_train, y_train, 
        epochs=epochs,
        batch_size=batchsize,
        verbose=1,
        validation_data=(x_val,y_val))

    losses =  history_callback.history["val_loss"]
    np.savetxt("losses" + str(i)+".txt",losses, fmt="%s")


