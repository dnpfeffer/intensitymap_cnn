import numpy as np
import pickle
import configparser
import sys
import tensorflow as tf

# arg parser to get command line arguments
import argparse

import lnn as lnn

### load in models
from models_to_load import *

np.random.seed(1337)

# read in the config file name
parser = argparse.ArgumentParser()
parser.add_argument('-fn', '--file_name', default='',
                            help='What file name to use for the config file')
args = parser.parse_args()
config_name = args.file_name

if config_name == '':
    sys.exit('Must give a config file name with -fn')

# where to store things
data_loc = '../data/'
predict_loc = data_loc + 'predictions/'

# load in the config file
config = configparser.ConfigParser()
config.read(data_loc + config_name + '.ini')
models = config.sections()

# defaults
defaults = lnn.get_config_info(config, models[0])

# start the work to load the validation maps
cur_map_loc = config[models[0]]['map_loc']
subFields = lnn.loadBaseFNames(cur_map_loc)

# set random seed so data is shuffeled the same way
# every time and then make the seed random
np.random.seed(1234)
np.random.shuffle(subFields)
np.random.seed()

# get map locations
valPer = 0.2
valPoint = int(len(subFields) * (1 - valPer))
base_val = [cur_map_loc + s for s in subFields[valPoint:]]

# debug
# debug_map_numbs = 2
# base_val = base_val[:debug_map_numbs]

# load in maps
data_val = np.array([lnn.utf8FileToMapAndLum(x, defaults['luminosity_byproduct'],
                                        defaults['threeD'],
                                        defaults['log_input'],
                                        defaults['make_map_noisy'],
                                        defaults['pre_pool'],
                                        defaults['pre_pool_z'],
                                        defaults['lum_func_size']) for x in base_val])

# start iterating for each model
for model_name in models:
    model_params = lnn.get_config_info(config, model_name)

    #print('Doing model {0}'.format(model_params['model_name']))
    #print(model_params)

    model = get_master_res_next(model_params['model_loc'], model_params['pix_x'],
                                model_params['pix_y'], model_params['pix_z'],
                                model_params['lum_func_size'],
                                extra_file_name='', file_name=model_params['model_name'],
                                dense_layer=model_params['dense_layer'], base_filters=model_params['base_filters'],
                                cardinality=model_params['cardinality'],
                                give_weights=model_params['give_weights'], use_bias=model_params['use_bias'])

    # dict to store everything
    stored_predictions = {}

    for data, base in zip(data_val, base_val):
        cur_map = data[0]
        cur_lum = data[1]

        #print('Testing on map {0}'.format(base))

        ### add gaussian noise
        if isinstance(model_params['make_map_noisy'], (tuple, list, np.ndarray)):
            #print('doing random noise')
            cur_map = lnn.add_noise_after_pool(cur_map, model_params['make_map_noisy'],
                                               model_params['pre_pool'], model_params['pre_pool_z'])
        elif model_params['make_map_noisy'] > 0:
            #print('doing normal noise')
            cur_map = lnn.add_noise_after_pool(cur_map, model_params['make_map_noisy'],
                                               model_params['pre_pool'], model_params['pre_pool_z'])

        # add in foregrounds
        if model_params['add_foregrounds']:
            #print('adding foregrounds')
            model_params_obj = lnn.ModelParams()
            model_params_obj.give_attributes(pre_pool=model_params['pre_pool'], pre_pool_z=model_params['pre_pool_z'])
            model_params_obj.clean_parser_data()
            model_params_obj.get_map_info(base + '_map.npz')

            cur_map = lnn.add_foreground_noise(cur_map, model_params_obj.pix_x, model_params_obj.pix_y,
                                               model_params_obj.omega_pix, model_params_obj.nu,
                                               model_params['pre_pool_z'],
                                               random_foreground_params=model_params['random_foreground_params'])

        # add in geometric noise
        if model_params['geometric_noise']:
            cur_map = lnn.add_geometric_noise_after_pool(cur_map,
                model_params['pre_pool'], model_params['pre_pool_z'],
                noise_fraction=1.0/22, max_noise=25)

        # do gaussian smoothing
        if model_params['gaussian_smoothing'] > 0:
            cur_map = lnn.apply_gaussian_smoothing(cur_map, model_params['gaussian_smoothing'])

        # don't try looking at things with under 500 sources at 10^6 L_sun
        if model_params['only_bright']:
            if cur_lum[36] < np.log10(5*10**2):
                pass

        ### make sure the output size is correct
        if model_params['lum_func_size'] is not None:
            if model_params['lum_func_size'] >= 1:
                #print('changing the size of the output luminosity function')
                cur_lum = cur_lum[:model_params['lum_func_size']]
            else:
                cur_lum = cur_lum[model_params['lum_func_size']:]

        # expand the dimensions of the map and luminosity byproduct to work with the tensor of the model
        base_map = np.expand_dims(cur_map, axis=0)
        base_lum = np.expand_dims(cur_lum, axis=0)

        ### make a prediction for the luminoisty byproduct for the given map
        cnn_lum = model.predict(tf.convert_to_tensor(base_map), steps=1)

        ### convert negative values to just 0
        cnn_lum[cnn_lum < 0] = 0

        # store results
        stored_predictions[base] = [cnn_lum[0], cur_lum]

    # save results after going through a model fully
    lnn.save_pickle(stored_predictions, predict_loc + model_params['file_name'])
