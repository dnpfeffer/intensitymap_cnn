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
from keras.backend.tensorflow_backend import set_session

########################
### Setup Learning Environment and set variables that one would want to change between runs
########################
### continue training an old network or start a new one
continue_training = False

### locations
mapLoc = '../../maps2/basic_Li/'
catLoc = '../../catalogues/'
modelLoc = '../../models2/'

### map info
numb_maps = 100
pix_x = 256
pix_y = 256

### luminosity function number of x-values
lum_func_size = 49

### file name for output
fileName = 'fullL_lum_4_layer_2D_model_long'
continue_training_model_loc = fileName + '_temp.hdf5'

### callBackPeriod for checkpoints and saving things midway through
callBackPeriod = 10

### number of maps to look at in a batch
batch_size = 40
steps_per_epoch = 40
epochs = 150

### number of gpus
numb_gpu = 4

### dropout rate for training
droprate = 0.2

### validation percent of data
valPer = 0.2

### variables for what we are training on
ThreeD = False
luminosity_byproduct = 'basicL'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 1.0

#sess = tf.Session(config=config)
#set_session(sess)

#########################
### Set Up the Model
#########################
make_model = True

if continue_training:
    try:
        model2 = keras.models.load_model(modelLoc + continue_training_model_loc)
        make_model = False
        fileName = continue_training_model_loc[:-5]
    except:
        print('Could not load model in {}.\nOpting to train a new model instead'.format(modelLoc + continue_training_model_loc))
        fileName = fileName + '2'

if make_model:
    model2 = keras.Sequential()

    ### convolutional layer
    model2.add(keras.layers.Conv2D(16, kernel_size=(5,5), strides=(1,1), activation='relu', input_shape=(pix_x, pix_y, numb_maps)))
    model2.add(keras.layers.BatchNormalization())
    ### use a convolution instead of a pool that acts like a pool
    model2.add(keras.layers.Conv2D(16, kernel_size=(2,2), strides=(2,2), activation='relu'))
    # model2.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model2.add(keras.layers.Dropout(droprate))

    ### convolutional layer
    model2.add(keras.layers.Conv2D(32, (5,5), activation='relu'))
    model2.add(keras.layers.BatchNormalization())
    ### use a convolution instead of a pool that acts like a pool
    model2.add(keras.layers.Conv2D(32, kernel_size=(2,2), strides=(2,2), activation='relu'))
    # model2.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model2.add(keras.layers.Dropout(droprate))

    ### convolutional layer
    model2.add(keras.layers.Conv2D(64, (5,5), activation='relu'))
    model2.add(keras.layers.BatchNormalization())
    ### use a convolution instead of a pool that acts like a pool
    model2.add(keras.layers.Conv2D(64, kernel_size=(2,2), strides=(2,2), activation='relu'))
    # model2.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model2.add(keras.layers.Dropout(droprate))

    ### convolutional layer
    model2.add(keras.layers.Conv2D(128, (5,5), activation='relu'))
    model2.add(keras.layers.BatchNormalization())
    ### use a convolution instead of a pool that acts like a pool
    model2.add(keras.layers.Conv2D(128, kernel_size=(2,2), strides=(2,2), activation='relu'))
    # model2.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model2.add(keras.layers.Dropout(droprate))

    # ### convolutional layer
    # model2.add(keras.layers.Conv2D(256, (5,5), activation='relu'))
    # model2.add(keras.layers.BatchNormalization())
    # ### use a convolution instead of a pool that acts like a pool
    # model2.add(keras.layers.Conv2D(256, kernel_size=(2,2), strides=(2,2), activation='relu'))
    # # model2.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    # model2.add(keras.layers.Dropout(droprate))

    ### flatten the network
    model2.add(keras.layers.Flatten())
    ### make a dense layer for the second to last step
    model2.add(keras.layers.Dense(1000, activation='relu'))
    ### finish it off with a dense layer with the number of output we want for our luminosity function
    model2.add(keras.layers.Dense(lum_func_size, activation='linear'))

    model2.compile(loss=keras.metrics.mse,
                  optimizer=keras.optimizers.SGD(),
                  metrics=[keras.metrics.mse])

model2.summary()

###########################
### Set up checkpoints to save the model
###########################
filePath = modelLoc + fileName + '_temp.hdf5'
checkpoint = keras.callbacks.ModelCheckpoint(filePath, monitor='loss', verbose=1, save_best_only=False, mode='auto', period=callBackPeriod)

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.mean_squared_error = []
        self.val_loss = []
        self.val_mean_squared_error = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.mean_squared_error.append(logs.get('mean_squared_error'))
        self.val_loss.append(logs.get('val_loss'))
        self.val_loss.append(logs.get('val_mean_squared_error'))

history = LossHistory()

callbacks_list = [checkpoint, history]
#callbacks_list = [checkpoint]

###########################
### Start Training the network
###########################
subFields = lnn.loadBaseFNames(mapLoc)
np.random.shuffle(subFields)

valPoint = int(len(subFields)*valPer)
base = [mapLoc + s for s in subFields[:valPoint]]
base_val = [mapLoc + s for s in subFields[valPoint:]]

dataset = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(base))
dataset = dataset.shuffle(buffer_size=len(base))
dataset = dataset.map(lambda item: tuple(tf.py_func(lnn.utf8FileToMapAndLum, [item, luminosity_byproduct, ThreeD], [tf.float64, tf.float64])))
dataset = dataset.repeat()
dataset = dataset.batch(batch_size)

dataset_val = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(base_val))
dataset_val = dataset_val.shuffle(buffer_size=len(base_val))
dataset_val = dataset_val.map(lambda item: tuple(tf.py_func(lnn.utf8FileToMapAndLum, [item, luminosity_byproduct, ThreeD], [tf.float64, tf.float64])))
dataset_val = dataset_val.repeat()
dataset_val = dataset_val.batch(batch_size)

multi_gpu_model2 = keras.utils.multi_gpu_model(model2, numb_gpu)
multi_gpu_model2.compile(loss=keras.metrics.mse,
                  optimizer=keras.optimizers.SGD(),
                  metrics=[keras.metrics.mse])
multi_gpu_model2.summary()

history = multi_gpu_model2.fit(dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, 
                        validation_data = dataset_val, validation_steps=3,
                        callbacks=callbacks_list, verbose=1)

model2.save(modelLoc + fileName +  '.hdf5')
model2.save_weights(modelLoc + fileName + '_weights.hdf5')

with open(modelLoc + fileName + '_history', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
