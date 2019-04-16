import numpy as np
import tensorflow as tf
import sys
import time

from skimage.measure import block_reduce

from .ioFuncs import *
# from .tools import *

# needed to save the map
from limlam_mocker import limlam_mocker as llm

# class containing info needed to train a CNN
class ModelParams:
    def __init__(self):
        # continue training an old network or start a new one
        self.continue_training = False

        # locations
        self.mapLoc = 'basic_Li'
        self.catLoc = 'catalogues'
        self.modelLoc = 'models3'

        # map info
        self.numb_maps = 100
        self.pix_x = 256
        self.pix_y = 256
        self.omega_pix = 0
        self.nu = []

        # luminosity function number of x-values
        self.lum_func_size = 49

        # file name for output
        self.fileName = 'basic_adam'
        self.continue_training_model_loc = self.fileName + '_temp.hdf5'

        # callBackPeriod for checkpoints and saving things midway through
        self.callBackPeriod = 10

        # number of maps to look at in a batch
        self.batch_size = 40
        self.steps_per_epoch = 100
        self.epochs = 100

        # number of gpus
        self.numb_gpu = 4

        # dropout rate for training
        self.droprate = 0.5

        # validation percent of data
        self.valPer = 0.2

        # number of maps to actually train on
        self.train_number = 0

        # number of layers
        self.numb_layers = 4

        # base number of filters
        self.base_filters = 64

        # use bias or not
        self.use_bias = True

        # kernel sizes
        self.kernel_size = 3
        self.pool_size = 2

        # final dense layer size
        self.dense_layer = 1000

        # size of pooling that should be done on maps before they are used
        # only pools in x and y direction, not z (actually redshift / frequency)
        self.pre_pool = 4
        self.pre_pool_z = 10

        # variables for what we are training on
        self.ThreeD = True
        self.luminosity_byproduct = 'log'
        self.log_input = True
        self.make_map_noisy = 0
        self.random_augmentation = False
        self.insert_foreground = False
        self.batchNorm_momentum = 0.99
        self.noise_lower = None
        self.noise_upper = None
        self.noise_limits = (None,None)
        self.random_foreground = False

        return()

    # setup command line parser arguments
    def setup_parser(self, parser):
        # add all of the arguments for command line parsing
        parser.add_argument('-fn', '--file_name', default='',
                            help='What file name to use')
        parser.add_argument('-c', '--continue_training',
                            type=self.str2bool, default=self.continue_training,
                            help='Should training start from scratch or continue')
        parser.add_argument('-bs', '--batch_size', type=int,
                            default=self.batch_size, help='Batch size to use')
        parser.add_argument('-spe', '--steps_per_epoch', type=int,
                            default=self.steps_per_epoch,
                            help='Number of steps(batches) per epoch')
        parser.add_argument('-e', '--epochs', type=int, default=self.epochs,
                            help='Number of epochs to train for')
        parser.add_argument('-g', '--numb_gpu', type=int,
                            default=self.numb_gpu, help='Number of GPUs to use')
        parser.add_argument('-d', '--dropout', type=float,
                            default=self.droprate, help='Dropout rate for each layer')
        parser.add_argument('-vp', '--validation_percent', type=float,
                            default=self.valPer,
                            help='Percent of data that is used for validation')
        parser.add_argument('-l', '--numb_layers', type=int,
                            default=self.numb_layers, help='Number of layers to use')
        parser.add_argument('-f', '--base_filters', type=int, default=self.base_filters,
                            help='Number of filters to use in the first layer')
        parser.add_argument('-3d', '--threeD', type=self.str2bool,
                            default=self.ThreeD, help='Use 3D convolutions or not')
        parser.add_argument('-lb', '--luminosity_byproduct',
                            default=self.luminosity_byproduct,
                            help='What luminosity function byproduct to train on')
        parser.add_argument('-li', '--log_input', type=self.str2bool,
                            default=self.log_input,
                            help='Take the log of the temperature map or not')
        parser.add_argument('-nm', '--make_map_noisy', type=float,
                            default=self.make_map_noisy,
                            help='Mean of the gaussian noise added to map in units of (micro K).  Not usable with --noise_upper or noise_lower.')
        parser.add_argument('-ks', '--kernel_size', type=int,
                            default=self.kernel_size, help='Kernel size of convolution')
        parser.add_argument('-mal', '--map_loc', default=self.mapLoc,
                            help='Location of maps')
        parser.add_argument('-cl', '--cat_loc', default=self.catLoc,
                            help='Location of catalogs')
        parser.add_argument('-mol', '--model_loc', default=self.modelLoc,
                            help='Location of models')
        parser.add_argument('-pp', '--pre_pool', type=int,
                            default=self.pre_pool, help='Kernel size for prepooling maps')
        parser.add_argument('-dl', '--dense_layer', type=int,
                            default=self.dense_layer, help='Size of the final dense layer')
        parser.add_argument('-lfs', '--lum_func_size', type=int,
                            default=self.lum_func_size,
                            help='Number of luminosity bins to fit for')
        parser.add_argument('-ub', '--use_bias', type=self.str2bool,
                            default=self.use_bias, help='Use the biases in the model')
        parser.add_argument('-ppz', '--pre_pool_z', type=int, default=self.pre_pool_z,
                            help='Kernel size for prepooling in z direction for maps')
        parser.add_argument('-tn', '--train_number', type=int, default=self.train_number,
                            help='Number of maps to actually train on.  0 is default and does 1-val_percent of the data.')
        parser.add_argument('-ra', '--random_augmentation', type=self.str2bool,
                            default=self.random_augmentation, help='Train with random augmentations or not')
        parser.add_argument('-if', '--insert_foreground', type=self.str2bool,
                            default=self.insert_foreground, help='Train with foregrounds or not')
        parser.add_argument('-bm', '--batchNorm_momentum', type=float,
                            default=self.batchNorm_momentum,
                            help='Momentum used by batch normalization')
        parser.add_argument('-nl', '--noise_lower', type=float,
                            default=self.noise_lower,
                            help='Lower limit of noise to consider in units of (micro K).  Must be used with --noise_upper, not usable with --make_map_noisy.')
        parser.add_argument('-nu', '--noise_upper', type=float,
                            default=self.noise_upper,
                            help='Upper limit of noise to consider in units of (micro K).  Must be used with --noise_lower, not usable with --make_map_noisy.')
        parser.add_argument('-rf', '--random_foreground', type=self.str2bool,
                            default=self.random_foreground, help='If foregrounds should be random or not')

        return()

    # read values from parser
    def read_parser(self, parser):
        # read in values for all of the argumnets
        args = parser.parse_args()
        self.fileName = args.file_name
        self.continue_training = args.continue_training
        self.batch_size = args.batch_size
        self.steps_per_epoch = args.steps_per_epoch
        self.epochs = args.epochs
        self.numb_gpu = args.numb_gpu
        self.droprate = args.dropout
        self.valPer = args.validation_percent
        self.numb_layers = args.numb_layers
        self.base_filters = args.base_filters
        self.ThreeD = args.threeD
        self.luminosity_byproduct = args.luminosity_byproduct
        self.log_input = args.log_input
        self.make_map_noisy = args.make_map_noisy
        self.kernel_size = args.kernel_size
        self.pre_pool = args.pre_pool
        self.dense_layer = args.dense_layer
        self.lum_func_size = args.lum_func_size
        self.use_bias = args.use_bias
        self.pre_pool_z = args.pre_pool_z
        self.train_number = args.train_number
        self.mapLoc = '../../maps2/{0}/'.format(args.map_loc)
        self.catLoc = '../../{0}/'.format(args.cat_loc)
        self.modelLoc = '../../{0}/'.format(args.model_loc)
        self.random_augmentation = args.random_augmentation
        self.insert_foreground = args.insert_foreground
        self.batchNorm_momentum = args.batchNorm_momentum
        self.noise_lower = args.noise_lower
        self.noise_upper = args.noise_upper
        self.random_foreground = args.random_foreground

        return()

    # set specific key value pairs for ModelParams attributes
    def give_attributes(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

    # do extra work on some parser parameters such as fixing pixel sizes
    def clean_parser_data(self):
        # make sure there is a file name
        if self.fileName == '':
            self.fileName = make_file_name(
                self.luminosity_byproduct, self.numb_layers, self.ThreeD, self.base_filters)

        # if there is a non-one pre_pool value, change the pix_x, pix_x and numb_maps accordingly
        if self.pix_x % self.pre_pool == 0:
            self.pix_x /= self.pre_pool
        else:
            exit_str = 'The pix_x value ({0}) must be divisible '\
                'by the pre-pooling kernel size ({1}).'
            sys.exit(
                exit_str.format(self.pix_x, self.pre_pool))

        if self.pix_y % self.pre_pool == 0:
            self.pix_y /= self.pre_pool
        else:
            exit_str = 'The pix_y value ({0}) must be divisible '\
                'by the pre-pooling kernel size ({1}).'
            sys.exit(
                exit_str.format(self.pix_y, self.pre_pool))

        if self.numb_maps % self.pre_pool_z == 0:
            self.numb_maps /= self.pre_pool_z
        else:
            exit_str = 'The numb_maps value ({0}) must be divisible '\
                'by the pre-pooling-z size ({1}).'
            sys.exit(
                exit_str.format(self.numb_maps, self.pre_pool_z))

        # make sure noise parameters are given correctly and don't comflict with each other
        if self.make_map_noisy > 0 and (self.noise_lower is not None or self.noise_upper is not None):
            sys.exit('make_map_noisy and noise_lower/noise_upper are being used together when they shouldn\'t.')

        if (self.noise_lower is not None and self.noise_upper is None) or (self.noise_lower is None and self.noise_upper is not None):
            sys.exit('Only one limit for the noise was given with noise_lower/noise_upper.  Both are required.')

        if self.noise_lower is not None and self.noise_upper is not None:
            if self.noise_upper < self.noise_lower:
                exit_str = 'noise_upper < noise_smaller ({0} < {1}).  It needs to be the other way.'
                sys.exit(exit_str.format(self.noise_upper, self.noise_lower))
            self.noise_limits = (self.noise_lower, self.noise_upper)
            self.make_map_noisy = 1

        return()

    # set basic map info about angular size and frequencies used for a given map
    def get_map_info(self, fName):
        # load mapa data
        data = loadMap_data(fName)

        # set map data into the data structure
        self.omega_pix = data['omega_pix'] * self.pre_pool**2
        self.nu = data['nu']

        return()

    # string to bool for the parser
    def str2bool(self, v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    # check to see if training should continue or not
    def continue_training_check(self):
        # return the loaded old model if training is to continue
        if self.continue_training:
            continue_count = get_model_iteration(self.fileName, model_loc=self.modelLoc)

            model = keras.models.load_model(self.modelLoc + self.fileName + '.hdf5')
            self.make_model = False
            self.continue_name = '_{0}'.format(continue_count)
            return(model)
        else:
            self.make_model = True
            self.continue_name = ''
            return(None)

    # set which type of loss to use
    def set_loss(self, losses):
        # choose which loss to use
        if self.luminosity_byproduct == 'log':
            # loss = keras.logcosh
            loss = losses.mse
        elif self.luminosity_byproduct == 'basic':
            loss = losses.msle
        elif self.luminosity_byproduct == 'basicL':
            loss = losses.msle
        elif self.luminosity_byproduct == 'numberCt':
            loss = losses.logcosh
        else:
            loss = losses.mse

        self.loss = loss

        return()

# make a file name from the given model information
def make_file_name(luminosity_byproduct, numb_layers, ThreeD, base_filters):
    # keep track of luminosity byproduct used
    if luminosity_byproduct == 'log':
        lb_string = 'log'
    elif luminosity_byproduct == 'basic':
        lb_string = 'full'
    elif luminosity_byproduct == 'basicL':
        lb_string = 'fullL'
    else:
        print('There should not be a way for someone to be in make_file_name without a valid luminosity_byproduct: {0}'.format(luminosity_byproduct))
        exit(0)

    # is this 3d or 2d (it should be 2d)
    if ThreeD:
        ThreeD_string = '3D'
    else:
        ThreeD_string = '2D'

    # full file name
    file_name = '{0}_lum_{1}_layer_{2}_{3}_filters_model'.format(lb_string, numb_layers, ThreeD_string, base_filters)

    return(file_name)

# setup checkpoints to be used for training to track loss and save periodically
def setup_checkpoints(model_params, callbacks):
    # save the model every so often based on model_params.callBackPeriod
    filePath = model_params.modelLoc + model_params.fileName + '_temp' + model_params.continue_name + '.hdf5'
    checkpoint = callbacks.ModelCheckpoint(
        filePath, monitor='loss', verbose=1, save_best_only=False,
        mode='auto', period=model_params.callBackPeriod)

    # keep track of loss information every epoch
    class LossHistory(callbacks.Callback):
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

    # package the two checkpoints together
    callbacks_list = [checkpoint, history]

    return(callbacks_list)

# load in data for training and package it for training
def setup_datasets(model_params):
    print('Starting to load data...')
    subFields = loadBaseFNames(model_params.mapLoc)

    # set random seed so data is shuffeled the same way
    # every time and then make the seed random
    np.random.seed(1234)
    np.random.shuffle(subFields)
    np.random.seed()

    # shuffle test and validation data
    valPoint = int(len(subFields) * (1 - model_params.valPer))

    # base file names for training data
    base = [model_params.mapLoc + s for s in subFields[:valPoint]]

    # only load a certain number of files if requested
    if model_params.train_number > 0:
        base = base[:model_params.train_number]

    # base file names for validation data
    base_val = [model_params.mapLoc + s for s in subFields[valPoint:]]

    # shuffle the order of the maps
    np.random.shuffle(base)
    np.random.shuffle(base_val)

    # put map information into our ModelParams object
    model_params.get_map_info(base[0] + '_map.npz')

    # make dataset used by tensorflow and modify maps with noise, foregrounds and etc
    dataset = make_dateset(model_params, base)

    # don't load the whole dataset if you are training on less maps then that
    # this was used for tests and shouldn't happen normally
    # also make the dataset for the validation data
    if model_params.train_number < len(base_val) and model_params.train_number > 0:
        dataset_val = make_dateset(model_params, base)
    else:
        dataset_val = make_dateset(model_params, base_val)

    print('Data fully loaded')
    return(dataset, dataset_val)

# load IMs, augment them with noise, foregrounds and etc and package into datasets for tensorflow
def make_dateset(model_params, base):
    # time how long it takes to load data
    start = time.time()
    data = np.array([utf8FileToMapAndLum(x, model_params.luminosity_byproduct,
                                        model_params.ThreeD,
                                        model_params.log_input,
                                        model_params.make_map_noisy,
                                        model_params.pre_pool,
                                        model_params.pre_pool_z,
                                        model_params.lum_func_size) for x in base])
    end = time.time()
    print('Time to load data:', end - start)

    # split the IMs and luminosity functions (features and labels)
    features = np.stack(data[:,0])
    labels = np.stack(data[:,1])

    # start making the dataset
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.shuffle(buffer_size=len(features))

    # do random augmentations if desired
    if model_params.random_augmentation == True:
        lumLogBinCents = loadLogBinCenters(base[0])
        dataset = dataset.map(lambda x, y:
                            tuple(tf.py_func(do_random_augmentation, [x,
                                                                y,
                                                                lumLogBinCents],
                                                                [tf.float64, tf.float64])))#,
                            #num_parallel_calls=24)

    # add in foregrounds if requested
    if model_params.insert_foreground == True:
        dataset = dataset.map(lambda x, y:
                            tuple(tf.py_func(add_foreground_noise_lum_wrapper, [x,
                                                                y,
                                                                model_params.pix_x,
                                                                model_params.pix_y,
                                                                model_params.omega_pix,
                                                                model_params.nu,
                                                                model_params.pre_pool_z,
                                                                0.1,
                                                                model_params.random_foreground],
                                                                [tf.float64, tf.float64])))#,
                            #num_parallel_calls=24)

    # this adds gaussian noise to pre-processed maps
    if model_params.make_map_noisy > 0:
        if model_params.noise_limits[0] is not None:
            noise = model_params.noise_limits
        else:
            noise = model_params.make_map_noisy

        dataset = dataset.map(lambda x, y:
                            tuple(tf.py_func(add_noise_after_pool_lum_wrapper, [x,
                                                                y,
                                                                noise,
                                                                model_params.pre_pool,
                                                                model_params.pre_pool_z,
                                                                0.1],
                                                                [tf.float64, tf.float64])))#,
                            #num_parallel_calls=24)

    # repeat dataset when it goes through all maps
    dataset = dataset.repeat()
    # set the batch size to what was given in ModelParams
    dataset = dataset.batch(model_params.batch_size)
    # prefetch 2 maps at a time to help speed up loading maps onto GPUs
    dataset = dataset.prefetch(2)

    return(dataset)

# function to convert a base name into the map map_cube and the wanted luminosity byproduct
def fileToMapAndLum(fName, lumByproduct='basic'):
    mapData, lumData = loadMapAndLum(fName, lumByproduct=lumByproduct)

    return(mapData, lumData)

# function to convert a utf-8 basename into the map map_cube and the luminosity byproduct
def utf8FileToMapAndLum(fName, lumByproduct='basic', ThreeD=False, log_input=False,
    make_map_noisy=0, pre_pool=1, pre_pool_z=1, lum_func_size=None):

    # be careful with strings sometime not being strings and needing to decode them
    if type(lumByproduct) is not str:
        lumByproduct = lumByproduct.decode("utf-8")
    if type(fName) is not str:
        fName = fName.decode("utf-8")

    # load file name and luminosity byproduct type to map and luminosity byproduct type
    mapData, lumData = fileToMapAndLum(fName, lumByproduct)

    # pool the data if it is requested
    if pre_pool > 1:
        if len(mapData) % pre_pool == 0:
            mapData = block_reduce(
                mapData, (pre_pool, pre_pool, pre_pool_z), np.sum)
        else:
            pass

    # use a log of some sorts to make the IM values span a single order of magnitude
    if log_input:
        # mapData = np.log10(mapData + 1e-6) + 6
        # mapData = log_map(mapData + 1e3) - 3
        mapData = log_modulus(mapData)

    # mean_map = np.mean(mapData)
    # std_map = np.std(mapData)

    # mapData = (mapData - mean_map)/std_map

    # decide if you want to use all luminosity values or a subset of them
    if lum_func_size is not None:
        if lum_func_size >= 1:
            # lumData = lumData[::lum_func_size]
            lumData = lumData[:lum_func_size]
        else:
            lumData = lumData[lum_func_size:]

    # give an extra dimension to the map if it is using 3d convolutions
    if ThreeD:
        # make sure to reshape the map data for the 3D convolutions
        mapData = mapData.reshape(len(mapData), len(
            mapData[0]), len(mapData[0][0]), 1)

    return(mapData, lumData)

# function to add noise to a map after it has already been pooled
def add_noise_after_pool(mapData, make_map_noisy, pre_pool, pre_pool_z, chance_to_not=0.0):

    # only add noise some fraction of the time of the time
    if np.random.rand() < (1-chance_to_not):
        shrink_size = pre_pool**2 * pre_pool_z

        # check if make_map_noisy is a scalar or list/tuple with min and max noise to conisder
        if isinstance(make_map_noisy, (tuple, list, np.ndarray)):
            if len(make_map_noisy) != 2:
                sys.exit("A noise list was given that has more then 2 values.  Please only give a scalar or a list with a min and max")
            else:
                make_map_noisy = np.random.uniform(make_map_noisy[0], make_map_noisy[1])
                # print(make_map_noisy)

        # Old way of doing noise (incorrect because noise can be negative)
        # # Use central limit theorem and get a single draw for the noise
        # # assume 160 draws is enough and my math is correct to get new mean and variance
        # new_mean = shrink_size / np.sqrt(2*np.pi) * make_map_noisy
        # new_std = np.sqrt(shrink_size / 2 * make_map_noisy**2 * (1 - 1/np.pi))
        # noise = np.maximum(np.random.normal(new_mean, new_std, mapData.shape), 0)

        # new way of doing noise (noise can be negative)
        new_std = np.sqrt(shrink_size) * make_map_noisy
        noise = np.random.normal(0, new_std, mapData.shape)

        # add the noise to the map correctly
        mapData = add_to_processed_map(mapData, noise)

    return(mapData)

# function to add noise to a map after it has already been pooled
# wraps around function that doesn't take the luminosity data
def add_noise_after_pool_lum_wrapper(mapData, luminosity, make_map_noisy, pre_pool, pre_pool_z, chance_to_not=0.0):
    mapData = add_noise_after_pool(mapData, make_map_noisy, pre_pool, pre_pool_z,
        chance_to_not=chance_to_not)

    return(mapData, luminosity)

# function to randomly augment the training data
# includes: setting everything to zero
#           making everything less bright
#           switch x and y axes
#           reverse x and y axes
def do_random_augmentation(mapData, lumData, lumLogBinCents):
    # have a small chance of setting maps to zero
    if np.random.rand() < 0.1:
        mapData, lumData = set_map_to_zero(mapData, lumData)
    # have chance to do other augmentations if the map isn't zero
    else:
        # lower luminosity
        if np.random.rand() < 0.1:
            mapData, lumData = scaleMapAndLum(mapData, lumData, lumLogBinCents)
        # # rotate x and y axes by 90 degrees
        # if np.random.rand() < 0.1:
        #     mapData = np.rot90(mapData)
        # # flip x-axis
        # if np.random.rand() < 0.1:
        #      mapData = np.flip(mapData, 1)
        # # flip y-axis
        # if np.random.rand() < 0.1:
        #      mapData = np.flip(mapData, 0)

    return(mapData, lumData)

# function to set mapData and lumData to zero
def set_map_to_zero(mapData, lumData):
    mapData = np.zeros(mapData.shape)
    lumData = np.zeros(lumData.shape)

    return(mapData, lumData)

# function to scale scale luminosities and the luminosity function
# to get more test data
def scaleMapAndLum(mapData, lumData, lumLogBinCents, bin_index_diff=-1):
    # if not given an index, randomly decide what luminosity to scale against
    if bin_index_diff == -1:
        bin_index_diff = int(np.random.poisson(5))

    # find ratio of luminosities for scaling and scale the map
    ratio = lumLogBinCents[0] / lumLogBinCents[0 + bin_index_diff]
    mapData = mapData * ratio

    # move over luminosity data to new bins to keep the correct amount of intensity at each luminosity
    lum_size = len(lumLogBinCents)
    for i, lum in enumerate(lumData):
        if i + bin_index_diff < lum_size:
            lumData[i] = lumData[i + bin_index_diff]
        else:
            lumData[i] = 0

    return(mapData, lumData)

# function to add foreground noise into an intensity map
def add_foreground_noise(mapData, Nx, Ny, omega_pix, nu, pre_pool_z, chance_to_not=0.0, random_foreground_params=False):

    # only add noise some fraction of the time of the time
    if np.random.rand() < (1-chance_to_not):
        # make foreground map
        foreground = makeFGcube(int(Nx), int(Ny), omega_pix, nu, random_foreground_params=random_foreground_params)

        # reduce map in z direction
        foreground = block_reduce(foreground, (1,1,pre_pool_z), np.sum)
        foreground = foreground.reshape(mapData.shape)

        # add foreground to current map
        mapData = add_to_processed_map(mapData, foreground)

    return(mapData)

# function to add foreground noise into an intensity map
# wraps around function that doesn't take the luminosity data
def add_foreground_noise_lum_wrapper(mapData, lumData, Nx, Ny, omega_pix, nu, pre_pool_z, chance_to_not=0.0, random_foreground_params=False):
    mapData = add_foreground_noise(mapData, Nx, Ny, omega_pix, nu, pre_pool_z, chance_to_not=chance_to_not, random_foreground_params=random_foreground_params)

    return(mapData, lumData)

# make foreground data cube
def makeFGcube(Nx,Ny,omega_pix,nu,N0=32.1,gamma=2.18,Smin=1,Smax=10**2.5,a0=0.39,sigma_a=0.33, random_foreground_params=False):
    '''
    Function which generates intensity cube from point-source foregrounds based
    on observations in arXiv 0912.2335

    INPUTS:
    Nx          Number of map pixels in the x direction
    Ny          Number of map pixels in the y direction
    omega_pix   Solid angle subtended by a single pixel
    nu          List of centers of frequency channels
    N0          Number of sources/deg^2 (fid: 32.1 +/- 3 deg^-2)
    gamma       Slope of source flux power law (fid: 2.18 +/- 0.12)
    Smin        Minimum 31 GHz flux (fid: 1 u.mJy)
    Smax        Maximum 31 GHz flux (fid: 10^2.5 u.mJy)
    a0          Mean spectral index (fid: 0.39)
    sigma_a     Variance of Gaussian spectral index dist. (fid: 0.33)
    '''

    if random_foreground_params:
        N0 = np.random.normal(32.1, 3)
        gamma = np.random.normal(2.18, 0.12)

    # Mean sources/pixel
    Nbar = N0*omega_pix

    # Array of source counts
    Nsrc = np.random.poisson(Nbar,(Nx,Ny))

    # 31GHz Flux PDF
    S0 = 1
    Sedge = ulogspace(Smin,Smax,1001)
    Sbin = binedge_to_binctr(Sedge)
    dS = np.diff(Sedge)

    dNdS = N0*(Sbin/S0)**-gamma
    PS = dNdS*dS/(dNdS*dS).sum()

    # Initialize foreground data cube
    TFG = np.zeros((Nx,Ny,nu.size))

    amounts = np.random.poisson(Nbar,Nx*Ny)
    sources = amounts[np.flatnonzero(amounts)]
    choices = [(x,y) for x in range(Nx) for y in range(Ny)]
    locs = np.random.permutation(choices)[:len(sources)]

    for source, loc in zip(sources, locs):
        alpha = np.random.normal(a0,sigma_a,source)
        S = np.random.choice(Sbin,size=source,p=PS)
        ii = loc[0]
        jj = loc[1]
        for nn in range(0,source):
                TFG[ii,jj,:] = TFG[ii,jj,:]+Tnu(S[nn],nu,alpha[nn],omega_pix)

    return(TFG)

# compute brightness temp for foregrounds
def Tnu(S31,nu,alpha,omega_pix):
    '''
    Computes brightness temperature as a function of frequency given a
    spectral index and an intensity at 31 GHz
    '''
    S = S31*(nu/(31))**-alpha

    c = 2.99792458*10**8
    kb = 1.38064852*10**-16
    sq_deg_to_str = (np.pi/180)**2
    Jy_to_erg_m2 = 10**-19

    conversion = 1/1000*Jy_to_erg_m2/(omega_pix*sq_deg_to_str) * c**2/(2 * kb * (nu.mean()*10**9)**2)*10**6

    return(S*conversion)

# compute logarithmically-spaced numpy array with linear xmin and xmax
def ulogspace(xmin,xmax,nx):
    '''
    Computes logarithmically-spaced numpy array between xmin and xmax with nx
    points.  This function calls the usual np.loglog but takes the linear
    values of xmin and xmax (where np.loglog requires the log of those limits)
    and allows the limits to have astropy units.  The output array will be a
    quantity with the same units as xmin and xmax
    '''

    return(np.logspace(np.log10(xmin),np.log10(xmax),nx))

# output centers of bins given their edges
def binedge_to_binctr(binedge):
    '''
    Outputs centers of histogram bins given their edges

    >>> Tedge = [1.,2.]*u.uK
    >>> print binedge_to_binctr(Tedge)
    [ 1.5] uK
    '''

    Nedge = binedge.size

    binctr = (binedge[0:Nedge-1]+binedge[1:Nedge])/2.

    return(binctr)

# function to take log of map and prevent taking the log of negative numbers
def log_map(cur_map):
    cur_map = np.log10(cur_map + 1e3) - 3
    return(cur_map)

# function to add a map to a post-processed map
def add_to_processed_map(old_map, new_map):
    # old_map = np.log10(np.power(10, old_map-6) + new_map) + 6
    # old_map = np.log10(np.power(10, old_map+3) + new_map) - 3

    old_map = log_modulus(undo_log_modulus(old_map) + new_map)

    return(old_map)

# special function to take log of positive and negative values and still work
# probably should use 1 instead of 1e-6 in the log and it wouldn't change anything
def log_modulus(cur_map):
    cur_map = np.sign(cur_map) * np.log10(np.abs(cur_map) + 1e-6)

    return(cur_map)

# undoes the log_modulus function
def undo_log_modulus(cur_map):
    cur_map = np.sign(cur_map) * (np.power(10, np.abs(cur_map)) - 1e-6)

    return(cur_map)

# add noise based on distance from edge of map
# after the map has been pooled already
def add_geometric_noise_after_pool(mapData, pre_pool, pre_pool_z, noise_fraction=1.0/22, max_noise=100):
    # get current shape of map and original shape
    pre_shape = mapData.shape
    shape = (pre_shape[0]*pre_pool, pre_shape[1]*pre_pool, pre_shape[2]*pre_pool_z)

    # make geometric noise map on original map
    noise_map = make_geometric_noise_map(shape, noise_fraction, max_noise)

    # pool the geometric noise map and add it to the original map
    noise_map = block_reduce(noise_map, (pre_pool,pre_pool,pre_pool_z), np.sum)
    noise_map = noise_map.reshape(mapData.shape)
    mapData = add_to_processed_map(mapData, noise_map)

    return(mapData)

# make geometric noise map
def make_geometric_noise_map(shape, noise_fraction=1.0/22, max_noise=100):
    # make an empty map of the correct size
    noise_map = np.zeros(shape)

    # set the max x and y pixel positions to worry about
    x_max = int(noise_fraction*noise_map.shape[0])
    y_max = int(noise_fraction*noise_map.shape[1])
    x_range = np.arange(-x_max, x_max)
    y_range = np.arange(-y_max, y_max)

    # loop through all x pixels that may matter and add noise
    for i in x_range:
        for j in range(noise_map.shape[1]):
            for k in range(noise_map.shape[2]):
                noise_map[i,j,k] = geometric_noise(i, j, noise_map.shape, noise_fraction, max_noise)

    # go through all y values and go to x values if they don't already have noise
    for i in range(noise_map.shape[0]):
        if i in x_range:
            continue
        for j in y_range:
            for k in range(noise_map.shape[2]):
                noise_map[i,j,k] = geometric_noise(i, j, noise_map.shape, noise_fraction, max_noise)

    return(noise_map)

# figure out noise in pixel based on location
def geometric_noise(i, j, shape, noise_fraction=1.0/22, max_noise=100):
    # consider negative values to be on the right or bottom of map
    if i < 0:
        i = shape[0]+i
    if j < 0:
        j = shape[1]+j

    # get fraction from side of map in x-direction
    x_frac = i/shape[0]

    # if the pixel is more then halfway consider 1-x_pix_location
    if x_frac > 0.5:
        x_frac = 1 - x_frac

    # get noise that should be in pixel due to distance from x-boundary
    x_noise = max(1.0-x_frac/noise_fraction, 0)
    x_noise = x_noise * max_noise

    # get fraction from side of map in y-direction
    y_frac = j/shape[1]

    # if the pixel is more then halfway consider 1-x_pix_location
    if y_frac > 0.5:
        y_frac = 1 - y_frac

    # get noise that should be in pixel due to distance from y-boundary
    y_noise = max(1.0-y_frac/noise_fraction, 0)
    y_noise = y_noise * max_noise

    # figure out which noise matters more and use it accordingly
    noise_frac = max(x_noise, y_noise)
    noise = np.random.normal(0, noise_frac * max_noise)

    return(noise)

# get model info from configuration file
def get_config_info(config, model_name):
    # setup config dictionary
    model_params = {}

    # read in data from config file
    model_params['file_name'] = config[model_name]['file_name']
    model_params['model_name'] = config[model_name]['model_name']
    model_params['model_loc'] = config[model_name]['model_loc']
    model_params['map_loc'] = config[model_name]['map_loc']
    model_params['cardinality'] = int(config[model_name]['cardinality'])
    model_params['give_weights'] = config[model_name].getboolean('give_weights')
    model_params['use_bias'] = config[model_name].getboolean('give_weights')
    model_params['pre_pool'] = int(config[model_name]['pre_pool'])
    model_params['pre_pool_z'] = int(config[model_name]['pre_pool_z'])
    model_params['pix_x'] = int(config[model_name]['pix_x'])
    model_params['pix_y'] = int(config[model_name]['pix_y'])
    model_params['pix_z'] = int(config[model_name]['pix_z'])
    model_params['dense_layer'] = int(config[model_name]['dense_layer'])
    model_params['base_filters'] = int(config[model_name]['base_filters'])
    model_params['lum_func_size'] = int(config[model_name]['lum_func_size'])

    model_params['luminosity_byproduct'] = config[model_name]['luminosity_byproduct']
    model_params['threeD'] = config[model_name].getboolean('threeD')
    model_params['log_input'] = config[model_name].getboolean('log_input')
    model_params['make_map_noisy'] = float(config[model_name]['make_map_noisy'])
    model_params['make_map_noisy2'] = float(config[model_name]['make_map_noisy2'])
    model_params['add_foregrounds'] = config[model_name].getboolean('add_foregrounds')
    model_params['random_foreground_params'] = config[model_name].getboolean('random_foreground_params')

    # manage if random noise is requested
    if model_params['make_map_noisy2'] != 0:
        model_params['make_map_noisy'] = (model_params['make_map_noisy'], model_params['make_map_noisy2'])

    return(model_params)
