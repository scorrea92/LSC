import keras
import get_data
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Get train and tes

epochs = 100
batchsize = 128
seed = 7
np.random.seed(seed)

def basic_model():
    model = Sequential()
    model.add(Dense(2048, input_shape=(76,), init='uniform', activation='relu'))
    model.add(Dense(1024, init='uniform', activation='relu'))
    model.add(Dense(512, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='linear'))
    model.summary()
    sgd=SGD(lr=0.01, decay=1e-6, momentum=0.9)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model


# Get Data
path_train = '../dataset_cajamar/Dataset_Salesforce_Predictive_Modelling_TRAIN.txt'
path_test = '../dataset_cajamar/Dataset_Salesforce_Predictive_Modelling_TEST.txt' 

x_train, y_train, x_val, y_val, test = get_data.import_data(path_train, path_test)

x_train = x_train.reshape(x_train.shape[0], 76)
x_val = x_val.reshape(x_val.shape[0], 76)

x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
# Tensorboard
tbCallBack = keras.callbacks.TensorBoard(log_dir='/tmp/keras_logs', write_graph=True)

# Compile and fit
model = basic_model()
model.fit(x_train, y_train, epochs=epochs, batch_size=batchsize,  verbose=1, validation_data={x_val,y_val}, callbacks=[tbCallBack])


