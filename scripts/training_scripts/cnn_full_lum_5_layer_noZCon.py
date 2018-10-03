#########################
### Base Imports
#########################
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle

### modify path to import limlam and lnn stuff
import sys
sys.path.append('../')

from limlam_mocker import limlam_mocker as llm
from limlam_mocker import params        as params
import lnn as lnn

import tensorflow as tf
from tensorflow import keras

########################
### Setup Learning Environment and set variables that one would want to change between runs
########################
### locations
mapLoc = '../../maps/test/'
catLoc = '../../catalogues/'
modelLoc = '../../models/'

### map info
numb_maps = 100
pix_x = 256
pix_y = 256

### luminosity function number of x-values
lum_func_size = 49

### file name for output
fileName = 'my_model_full_lum_gpu_5_layer_noZCon'

### callBackPeriod for checkpoints and saving things midway through
callBackPeriod = 10

### number of maps to look at in a batch
batch_size = 2
steps_per_epoch = 200
epochs = 100

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.4

#########################
### Set Up the Model
#########################
model2 = keras.Sequential()

### convolutional layer
model2.add(keras.layers.Conv3D(32, kernel_size=(5,5,1), strides=(1,1,1), activation='relu', input_shape=(pix_x, pix_y, numb_maps, 1)))
### use a convolution instead of a pool that acts like a pool
#model2.add(keras.layers.Conv3D(32, kernel_size=(2,2,2), strides=(2,2,2), activation='relu'))
model2.add(keras.layers.MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2)))

### convolutional layer
model2.add(keras.layers.Conv3D(64, (5,5,1), activation='relu'))
### use a convolution instead of a pool that acts like a pool
#model2.add(keras.layers.Conv3D(64, kernel_size=(2,2,2), strides=(2,2,2), activation='relu'))
model2.add(keras.layers.MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2)))

### convolutional layer
model2.add(keras.layers.Conv3D(128, (5,5,1), activation='relu'))
### use a convolution instead of a pool that acts like a pool
#model2.add(keras.layers.Conv3D(64, kernel_size=(2,2,2), strides=(2,2,2), activation='relu'))
model2.add(keras.layers.MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2)))

### convolutional layer
model2.add(keras.layers.Conv3D(256, (5,5,1), activation='relu'))
### use a convolution instead of a pool that acts like a pool
#model2.add(keras.layers.Conv3D(64, kernel_size=(2,2,2), strides=(2,2,2), activation='relu'))
model2.add(keras.layers.MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2)))

### convolutional layer
model2.add(keras.layers.Conv3D(512, (5,5,1), activation='relu'))
### use a convolution instead of a pool that acts like a pool
#model2.add(keras.layers.Conv3D(64, kernel_size=(2,2,2), strides=(2,2,2), activation='relu'))
model2.add(keras.layers.MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2)))

### flatten the network
model2.add(keras.layers.Flatten())
### make a dense layer for the second to last step
model2.add(keras.layers.Dense(1000, activation='relu'))
### finish it off with a dense layer with the number of output we want for our luminosity function
model2.add(keras.layers.Dense(lum_func_size, activation='linear'))

model2.compile(loss=keras.losses.msle,
              optimizer=keras.optimizers.SGD(),
              metrics=[keras.metrics.mse])

###########################
### Set up checkpoints to save the model
###########################
filePath = modelLoc + fileName + '_temp.hdf5'
checkpoint = keras.callbacks.ModelCheckpoint(filePath, monitor='val_loss', verbose=1, save_best_only=False, mode='auto', period=callBackPeriod)

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.mean_squared_error = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.mean_squared_error.append(logs.get('mean_squared_error'))

history = LossHistory()

callbacks_list = [checkpoint, history]
#callbacks_list = [checkpoint]

###########################
### Start Training the network
###########################
subFields = lnn.loadBaseFNames(mapLoc)
base = [mapLoc + s for s in subFields]

dataset = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(base))
dataset = dataset.shuffle(buffer_size=len(base))
dataset = dataset.map(lambda item: tuple(tf.py_func(lnn.utf8FileToMapAndLum3D, [item], [tf.float64, tf.float64])))
dataset = dataset.repeat()
dataset = dataset.batch(batch_size)

history = model2.fit(dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks_list, verbose=1)

model2.save(modelLoc + fileName +  '.hdf5')

with open(modelLoc + fileName + '_history', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)







