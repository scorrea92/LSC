import keras
import get_data
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras import regularizers, optimizers
from keras.layers.normalization import BatchNormalization as BN
from keras.layers import GaussianNoise as GN
from sklearn import preprocessing
from sklearn.externals import joblib 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.callbacks import LearningRateScheduler

# Basic NN config and reproduction seed
epochs = 100
batchsize = 128
seed = 7
np.random.seed(seed)

def step_decay(epoch): # 0.5 0.1 0.01 0.001
    if epoch<30:
        lrate = 0.5
    elif epoch<=50:
        lrate = 0.1
    elif epoch<=70:
        lrate = 0.01
    else:
        lrate = 0.01
    return lrate

def basic_model(): # 1024 512
    model = Sequential()
    model.add(Dense(2048, input_shape=(76,)))
    model.add(BN())
    model.add(Activation('relu'))
    
    model.add(Dense(1024))
    model.add(BN())
    model.add(GN(0.1))
    model.add(Activation('relu'))

    model.add(Dense(512))
    model.add(BN())
    model.add(GN(0.1))
    model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('relu'))
    model.summary()

    adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(loss='mape', optimizer=adam, metrics=['mse'])
    return model

# Get Data
path_train = '../dataset_cajamar/Dataset_Salesforce_Predictive_Modelling_TRAIN.txt'
path_test = '../dataset_cajamar/Dataset_Salesforce_Predictive_Modelling_TEST.txt' 

x_train, y_train, x_val, y_val, test = get_data.import_data(path_train, path_test)

x_train = x_train.reshape(x_train.shape[0], 76)
x_val = x_val.reshape(x_val.shape[0], 76)

y_train = y_train.reshape(y_train.shape[0], 1)
y_val = y_val.reshape(y_val.shape[0], 1)

x_train = x_train.astype('float32')
x_val = x_val.astype('float32')

y_train = y_train.astype('float32')
y_val = y_val.astype('float32')

# LRA
lrate = LearningRateScheduler(step_decay)

# Compile and fit
model = basic_model()
model.fit(x_train, y_train, 
        epochs=epochs,
        batch_size=batchsize,
        verbose=1,
        validation_data=(x_val,y_val),
        callbacks=[lrate])

predictions = model.predict(x_val)
error = np.absolute(predictions - y_val)

print(error)
print(error.min())
print(error.max())
print(error.mean())
print(error.std())
    


