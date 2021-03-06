from time import time

import get_data
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import GaussianNoise as GN
from keras.layers.normalization import BatchNormalization as BN
from keras.callbacks import LearningRateScheduler as LRS
from keras.optimizers import SGD
from sklearn.preprocessing import MinMaxScaler

# learning rate schedule
def step_decay(epoch):

  if epoch > 30:
      lrate = 0.001
  elif epoch > 50:
      lrate = 0.001
  else:
      lrate = 0.1

  return lrate

def float_adq_to_categorical(value):
    # 2 gaussians
    if value < 5000.0:
        return 0
    elif value < 40000.0:
        return 1
    else:
        return 2

def cmodel():
    model = Sequential()

    # Dense 1
    model.add(Dense(1024, input_shape=(90,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(Dense(1024))
    model.add(BN())
    model.add(GN(0.1))
    model.add(Activation('relu'))

    model.add(Dense(512))
    model.add(BN())
    model.add(GN(0.1))
    model.add(Activation('relu'))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.summary()

    sgd = SGD(lr=0.0, momentum=0.9, decay=0.0, nesterov=False)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model


# Parameters
batch_size = 128
num_classes = 3
epochs = 50

seed = 7
np.random.seed(seed)

# Get Data
path_train = 'data_train.txt'
path_test = 'data_test.txt'

x_train, y_train, x_val, y_val, test = get_data.import_nopca(path_train, path_test)

# Transform y to categorical
y_train = [float_adq_to_categorical(y.item(0)) for y in y_train]
y_val = [float_adq_to_categorical(y.item(0)) for y in y_val]

y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)

print("Paso_2 : categorical value")
# np.savetxt("ydata_train_pca.txt",y_train, fmt='%i', delimiter=",")
print(np.unique(y_train, axis=0))

# Scale X
scaler = MinMaxScaler()
scaler.fit(x_train)
scaler.fit(x_val)
scaler.transform(x_train)
scaler.transform(x_val)

#print(x_train.min())
#print(x_train.max())

# x_train = x_train[:, 38:]
# x_val = x_val[:, 38:]

print("x train shape")
print(x_train.shape)

x_train = x_train.reshape(x_train.shape[0], 90)
x_val = x_val.reshape(x_val.shape[0], 90)

x_train = x_train.astype('float32')
x_val = x_val.astype('float32')


# learning schedule callback
lrate = LRS(step_decay)
callbacks_list = [lrate]

#callbacks = [TensorBoard(log_dir="logs/classifier/{}".format(time()), write_graph=True)]

model = cmodel()
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_val, y_val),
                    shuffle=True,
                    callbacks=callbacks_list)

scores = model.evaluate(x_val, y_val, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
