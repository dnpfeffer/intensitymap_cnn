#########################
### Base Imports
#########################
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle

### arg parser to get command line arguments
import argparse

### modify path to import limlam and lnn stuff
import sys
sys.path.append('../')

from limlam_mocker import limlam_mocker as llm
from limlam_mocker import params        as params
import lnn as lnn

### load in models
from models_to_load import *

import tensorflow as tf
from tensorflow import keras
from keras.backend.tensorflow_backend import set_session

########################
### Setup Learning Environment and set variables that one would want to change between runs
########################
### continue training an old network or start a new one
continue_training = False

### locations
mapLoc = 'basic_Li'
catLoc = 'catalogues'
modelLoc = 'models3'

### map info
numb_maps = 100
pix_x = 256
pix_y = 256

### luminosity function number of x-values
lum_func_size = 49

### file name for output
fileName = 'log_lum_5_layer_2D_model_long'
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

### number of layers
numb_layers = 4

### base number of filters
base_filters = 16

### kernel sizes
kernel_size = 5
pool_size = 2


### variables for what we are training on
ThreeD = False
luminosity_byproduct = 'log'
log_input = False
make_map_noisy = 0

### handle the argument parsing
parser = argparse.ArgumentParser()

### handle booleans
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

### add all of the arguments
parser.add_argument('-fn', '--file_name', default='', help='What file name to use')
parser.add_argument('-c', '--continue_training', type=lnn.str2bool, default=continue_training, help='Should training start from scratch or continue from before')
parser.add_argument('-bs', '--batch_size', type=int, default=batch_size, help='Batch size to use')
parser.add_argument('-spe', '--steps_per_epoch', type=int, default=steps_per_epoch, help='Number of steps(batches) per epoch')
parser.add_argument('-e', '--epochs', type=int, default=epochs, help='Number of epochs to train for')
parser.add_argument('-g', '--numb_gpu', type=int, default=numb_gpu, help='Number of GPUs to use')
parser.add_argument('-d', '--dropout', type=float, default=droprate, help='Dropout rate for each layer')
parser.add_argument('-vp', '--validation_percent', type=float, default=valPer, help='Percent of data that is used for validation')
parser.add_argument('-l', '--numb_layers', type=int, default=numb_layers, help='Number of layers to use')
parser.add_argument('-f', '--base_filters', type=int, default=base_filters, help='Number of filters to use in the first layer')
parser.add_argument('-3d', '--threeD', type=lnn.str2bool, default=ThreeD, help='Use 3D convolutions or not')
parser.add_argument('-lb', '--luminosity_byproduct', default=luminosity_byproduct, help='What luminosity function byproduct to train on')
parser.add_argument('-li', '--log_input', type=lnn.str2bool, default=log_input, help='Take the log of the temperature map or not')
parser.add_argument('-nm', '--make_map_noisy', type=float, default=make_map_noisy, help='Number of filters to use in the first layer')
parser.add_argument('-ks', '--kernel_size', type=int, default=kernel_size, help='Kernel size of convolution')
parser.add_argument('-mal', '--map_loc', default=mapLoc, help='Location of maps')
parser.add_argument('-cl', '--cat_loc', default=catLoc, help='Location of catalogs')
parser.add_argument('-mol', '--model_loc', default=modelLoc, help='Location of models')


### read in values for all of the argumnets
args = parser.parse_args()
fileName = args.file_name
continue_training = args.continue_training
batch_size = args.batch_size
steps_per_epoch = args.steps_per_epoch
epochs = args.epochs
numb_gpu = args.numb_gpu
droprate = args.dropout
valPer = args.validation_percent
numb_layers = args.numb_layers
base_filters = args.base_filters
ThreeD = args.threeD
luminosity_byproduct = args.luminosity_byproduct
log_input = args.log_input
make_map_noisy = args.make_map_noisy
kernel_size = args.kernel_size

if fileName == '':
    fileName = lnn.make_file_name(luminosity_byproduct, numb_layers, ThreeD, base_filters)

mapLoc = '../../maps2/{0}/'.format(args.map_loc)
catLoc = '../../{0}/'.format(args.cat_loc)
modelLoc = '../../{0}/'.format(args.model_loc)

### set up how much memory the gpus use
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 1.0

#sess = tf.Session(config=config)
#set_session(sess)

#########################
### Set Up the Model
#########################

if continue_training:
    continue_count = lnn.get_model_iteration(fileName, model_loc=modelLoc)

    master = keras.models.load_model(modelLoc + fileName + '.hdf5')
    make_model = False
    continue_name = '_{0}'.format(continue_count)
else:
    continue_name = ''

### choose which loss to use
if luminosity_byproduct == 'log':
    loss = keras.losses.logcosh
elif luminosity_byproduct == 'basic':
    loss = keras.losses.msle
elif luminosity_byproduct == 'basicL':
    loss = keras.losses.msle
elif luminosity_byproduct == 'numberCt':
    loss = keras.losses.logcosh
else:
    loss = keras.losses.mse

model2 = get_master_2(modelLoc, pix_x, pix_y, numb_maps, lum_func_size,
                train_number=0,
                droprate=droprate, numb_layers=numb_layers,
                base_filters=base_filters, threeD=ThreeD,
                luminosity_byproduct='luminosity_byproduct, kernel_size=kernel_size)

# model2.summary()

###########################
### Set up checkpoints to save the model
###########################
filePath = modelLoc + fileName + '_temp' + continue_name  + '.hdf5'
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

##########################
## Start Training the network
##########################
subFields = lnn.loadBaseFNames(mapLoc)

### set random seed so data is shuffeled the same way every time and then make the seed random
np.random.seed(1234)
np.random.shuffle(subFields)
np.random.seed()

### shuffle  test and validation data
valPoint = int(len(subFields)*(1-valPer))
base = [mapLoc + s for s in subFields[:valPoint]]
base_val = [mapLoc + s for s in subFields[valPoint:]]
np.random.shuffle(base)
np.random.shuffle(base_val)


dataset = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(base))
dataset = dataset.shuffle(buffer_size=len(base))
dataset = dataset.map(lambda item: tuple(tf.py_func(lnn.utf8FileToMapAndLum, [item, luminosity_byproduct, ThreeD, log_input, make_map_noisy], [tf.float64, tf.float64])))
dataset = dataset.repeat()
dataset = dataset.batch(batch_size)

dataset_val = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(base_val))
dataset_val = dataset_val.shuffle(buffer_size=len(base_val))
dataset_val = dataset_val.map(lambda item: tuple(tf.py_func(lnn.utf8FileToMapAndLum, [item, luminosity_byproduct, ThreeD, log_input, make_map_noisy], [tf.float64, tf.float64])))
dataset_val = dataset_val.repeat()
dataset_val = dataset_val.batch(batch_size)

multi_gpu_model2 = keras.utils.multi_gpu_model(model2, numb_gpu)
multi_gpu_model2.compile(loss=loss,
                  optimizer=keras.optimizers.SGD(),
                  metrics=[keras.metrics.mse])
# multi_gpu_model2.summary()

history = multi_gpu_model2.fit(dataset, epochs=epochs, steps_per_epoch=steps_per_epoch,
                        validation_data = dataset_val, validation_steps=3,
                        callbacks=callbacks_list, verbose=1)

model2.save(modelLoc + fileName + continue_name + '.hdf5')
model2.save_weights(modelLoc + fileName + '_weights' + continue_name + '.hdf5')

with open(modelLoc + fileName + '_history' + continue_name, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
