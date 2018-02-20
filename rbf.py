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
from keras.optimizers import RMSprop
from rbflayer import RBFLayer, InitCentersRandom

# Basic NN config and reproduction seed
epochs = 10
batchsize = 64
seed = 7
np.random.seed(seed)

def step_decay(epoch):
    if epoch<30:
        lrate = 0.5
    elif epoch<=50:
        lrate = 0.1
    elif epoch<=70:
        lrate = 0.01
    else:
        lrate = 0.001
    return lrate

def data(path_train, path_test):
    x_train, y_train, x_val, y_val, test = get_data.import_data(path_train, path_test)
    
    x_train = x_train.reshape(x_train.shape[0], 76)
    x_val = x_val.reshape(x_val.shape[0], 76)

    y_train = y_train.reshape(y_train.shape[0], 1)
    y_val = y_val.reshape(y_val.shape[0], 1)

    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')

    y_train = y_train.astype('float32')
    y_val = y_val.astype('float32')

    print(y_val.min())
    print(y_val.max())
    print(y_val.mean())
    print(y_val.std())

    print(y_train.min())
    print(y_train.max())
    print(y_train.mean())
    print(y_train.std())

    y_val_nostandard = y_val
    y_train_nostandard = y_train

    scaler = StandardScaler()
    scaler.fit(y_train)

    y_train = scaler.transform(y_train)
    y_val = scaler.transform(y_val)

    return x_train, y_train, x_val, y_val, test, scaler, y_val_nostandard, y_train_nostandard

# Get Data
path_train = '../dataset_cajamar/Dataset_Salesforce_Predictive_Modelling_TRAIN.txt'
path_test = '../dataset_cajamar/Dataset_Salesforce_Predictive_Modelling_TEST.txt' 

x_train, y_train, x_val, y_val, test, scaler, y_val_nostandard, y_train_nostandard = data(path_train, path_test)

model = Sequential()
rbflayer = RBFLayer(10,
                    initializer=InitCentersRandom(x_train), 
                    betas=2.0,
                    input_shape=(76,))
model.add(rbflayer)

model.add(Dense(512))
model.add(BN())
model.add(GN(0.3))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('relu'))

model.compile(loss='mape', optimizer=RMSprop(), metrics=['mse'])

model.fit(x_train, y_train,
            batch_size=batchsize,
            epochs=epochs,
            validation_data=(x_val,y_val),
            verbose=1)

predictions = model.predict(x_val)
predictions = scaler.inverse_transform(predictions)
error = np.absolute(predictions - y_val_nostandard)
mape = error/predictions

print("Val")
print(error)
print(error.min())
print(error.max())
print(error.mean())
print(error.std())
print("mape", mape.mean())


predictions = model.predict(x_train)
predictions = scaler.inverse_transform(predictions)
error = np.absolute(predictions - y_train_nostandard)

print("Train")
print(error)
print(error.min())
print(error.max())
print(error.mean())
print(error.std())


    