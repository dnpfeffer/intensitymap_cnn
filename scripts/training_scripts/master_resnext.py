#########################
# Base Imports
#########################
import matplotlib as mpl
mpl.use('Agg')
# import matplotlib.pyplot as plt
import numpy as np
import pickle

# arg parser to get command line arguments
import argparse

# modify path to import limlam and lnn stuff
import sys
sys.path.append('../')

# from limlam_mocker import limlam_mocker as llm
# from limlam_mocker import params as params
import lnn as lnn

# load in models
from models_to_load import *

import tensorflow as tf
from tensorflow import keras
from keras.backend.tensorflow_backend import set_session

import time

########################
# Setup Learning Environment and set variables that
# one would want to change between runs
########################
parser = argparse.ArgumentParser()
model_params = lnn.ModelParams()
model_params.setup_parser(parser)
model_params.read_parser(parser)
model_params.clean_parser_data()

#########################
# Set Up the Model
#########################
### load old model if required
model2 = model_params.continue_training_check()

### get loss function
model_params.set_loss(keras.losses)

if model2 is None:
    model2 = get_master_res_next2(model_params)

# model2.summary()

###########################
# Set up checkpoints to save the model
###########################
callbacks_list = lnn.setup_checkpoints(model_params, keras.callbacks)

##########################
# Start Training the network
##########################
dataset, dataset_val = lnn.setup_datasets(model_params)

print('-------------------------')
print(dataset)
print(model_params.batch_size, model_params.epochs, model_params.steps_per_epoch)
print('-------------------------')

multi_gpu_model2 = keras.utils.multi_gpu_model(model2, 4)
multi_gpu_model2.compile(loss=model_params.loss,
                         optimizer=keras.optimizers.Adam(),
                         metrics=[keras.metrics.mse])
# multi_gpu_model2.summary()

history = multi_gpu_model2.fit(dataset, epochs=model_params.epochs,
                               steps_per_epoch=model_params.steps_per_epoch,
                               validation_data=dataset_val, validation_steps=1,
                               callbacks=callbacks_list, verbose=1)

model2.save(model_params.modelLoc + model_params.fileName + model_params.continue_name + '.hdf5')
model2.save_weights(model_params.modelLoc + model_params.fileName + '_weights' + model_params.continue_name + '.hdf5')

with open(model_params.modelLoc + model_params.fileName + '_history' + model_params.continue_name, 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
