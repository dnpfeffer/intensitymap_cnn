import numpy as np
import tensorflow as tf
import time

from skimage.measure import block_reduce

from .ioFuncs import *
from .tools import *

# needed to save the map
from limlam_mocker import limlam_mocker as llm

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

    def setup_parser(self, parser):
        # add all of the arguments
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
                            help='Mean of the gaussian noise added to map in units of (micro K)')
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

    def clean_parser_data(self):
        if self.fileName == '':
            self.fileName = lnn.make_file_name(
                self.luminosity_byproduct, self.numb_layers, self.ThreeD, self.base_filters)

        # if there is a non-one pre_pool value, change the pix_x and pix_x accordingly
        if self.pix_x % self.pre_pool == 0:
            self.pix_x /= self.pre_pool
        else:
            exit_str = 'The pix_x value ({0}) must be divisible '\
                'by the pre-pooling kernel size ({1})'
            sys.exit(
                exit_str.format(self.pix_x, self.pre_pool))

        if self.pix_y % self.pre_pool == 0:
            self.pix_y /= self.pre_pool
        else:
            exit_str = 'The pix_y value ({0}) must be divisible '\
                'by the pre-pooling kernel size ({1})'
            sys.exit(
                exit_str.format(self.pix_y, self.pre_pool))

        if self.numb_maps % self.pre_pool_z == 0:
            self.numb_maps /= self.pre_pool_z
        else:
            exit_str = 'The numb_maps value ({0}) must be divisible '\
                'by the pre-pooling-z size ({1})'
            sys.exit(
                exit_str.format(self.numb_maps, self.pre_pool_z))

    def str2bool(self, v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    def continue_training_check(self):
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

def setup_checkpoints(model_params, callbacks):
    filePath = model_params.modelLoc + model_params.fileName + '_temp' + model_params.continue_name + '.hdf5'
    checkpoint = callbacks.ModelCheckpoint(
        filePath, monitor='loss', verbose=1, save_best_only=False,
        mode='auto', period=model_params.callBackPeriod)


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

    callbacks_list = [checkpoint, history]

def setup_datasets(model_params):
    print('Starting to load data...')
    subFields = loadBaseFNames(model_params.mapLoc)

    # set random seed so data is shuffeled the same way
    # every time and then make the seed random
    np.random.seed(1234)
    np.random.shuffle(subFields)
    np.random.seed()

    # shuffle  test and validation data
    valPoint = int(len(subFields) * (1 - model_params.valPer))
    base = [model_params.mapLoc + s for s in subFields[:valPoint]]

    if model_params.train_number > 0:
        base = base[:model_params.train_number]

    base_val = [model_params.mapLoc + s for s in subFields[valPoint:]]
    np.random.shuffle(base)
    np.random.shuffle(base_val)

    dataset = make_dateset(model_params, base)

    if model_params.train_number < len(base_val) and model_params.train_number > 0:
        dataset_val = make_dateset(model_params, base)
    else:
        dataset_val = make_dateset(model_params, base_val)

    print('Data fully loaded')
    return(dataset, dataset_val)

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

    features = np.stack(data[:,0])
    labels = np.stack(data[:,1])

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
    # this adds gaussian noise to pre-processed maps so that doesn't mean anything
    if model_params.make_map_noisy > 0:
        dataset = dataset.map(lambda x, y:
                            tuple(tf.py_func(add_noise_after_pool, [x,
                                                                model_params.make_map_noisy,
                                                                model_params.pre_pool,
                                                                model_params.pre_pool_z,
                                                                y],
                                                                [tf.float64, tf.float64])))#,
                            #num_parallel_calls=24)
    dataset = dataset.repeat()
    dataset = dataset.batch(model_params.batch_size)
    dataset = dataset.prefetch(2)

    return(dataset)

# function to convert a basename into the map map_cube and the wanted luminosity byproduct
def fileToMapAndLum(fName, lumByproduct='basic'):
    mapData, lumData = loadMapAndLum(fName, lumByproduct=lumByproduct)
    # mapData = maps['map_cube']
    # lumData = lumFuncByproduct(lumInfo, lumByproduct)

    return(mapData, lumData)

# function to convert a utf-8 basename into the map map_cube and the luminosity byproduct
def utf8FileToMapAndLum(fName, lumByproduct='basic', ThreeD=False, log_input=False,
                        make_map_noisy=0, pre_pool=1, pre_pool_z=1, lum_func_size=None):

    if type(lumByproduct) is not str:
        lumByproduct = lumByproduct.decode("utf-8")
    if type(fName) is not str:
        fName = fName.decode("utf-8")
    mapData, lumData = fileToMapAndLum(fName, lumByproduct)

    # print(fName)
    # print(mapData)
    # print(lumData)

    #########################
    # very temp thing that needs to be done correctly later
    # make some maps have only zeros and the lum func be zero as well
    # randomly scale things that aren't zero
    ########################
    # if np.random.rand() < 0.1:
    #     mapData = np.zeros(mapData.shape)
    #     lumData = np.zeros(lumData.shape)
    #     # pass
    # else:
    #     lumLogBinCents = loadLogBinCenters(fName.decode('utf-8'))
    #     # bin_index_diff = np.random.randint(0, len(lumLogBinCents)-1)
    #     bin_index_diff = int(np.random.poisson(5))
    #     mapData, lumData = scaleMapAndLum(
    #         mapData, lumData, lumLogBinCents, bin_index_diff)
    #     # pass
    ########################
    #######################


    # add gaussian noise, but make sure it is positive valued
    # if make_map_noisy > 0:
    #     mapData = mapData + \
    #         np.maximum(np.random.normal(0, make_map_noisy, mapData.shape), 0)

    if pre_pool > 1:
        if len(mapData) % pre_pool == 0:
            mapData = block_reduce(
                mapData, (pre_pool, pre_pool, pre_pool_z), np.sum)
        else:
            pass

    if log_input:
        mapData = np.log10(mapData + 1e-6)

        # mapData -= (np.min(mapData))
        mapData -= (-6)

    # mean_map = np.mean(mapData)
    # std_map = np.std(mapData)

    # mapData = (mapData - mean_map)/std_map

    if lum_func_size is not None:
        if lum_func_size >= 1:
            # lumData = lumData[::lum_func_size]
            lumData = lumData[:lum_func_size]
        else:
            lumData = lumData[lum_func_size:]

    if ThreeD:
        # make sure to reshape the map data for the 3D convolutions
        mapData = mapData.reshape(len(mapData), len(
            mapData[0]), len(mapData[0][0]), 1)

    return(mapData, lumData)

# function to add noise to a map after it has already been pooled
def add_noise_after_pool(mapData, make_maep_noisy, pre_pool, pre_pool_z, luminosity=None):

    # only add noise 90% of the time
    if np.random.rand() < 0.9:
        shrink_size = pre_pool**2 * pre_pool_z

        # Use central limit theorem and get a single draw for the noise
        # assume 160 draws is enough and my math is correct to get new mean and variance
        new_mean = shrink_size / np.sqrt(2*np.pi) * make_maep_noisy
        new_std = np.sqrt(shrink_size / 2 * make_maep_noisy**2 * (1 - 1/np.pi))
        noise = np.maximum(np.random.normal(new_mean, new_std, mapData.shape), 0)

        # add the noise to the map correctly
        mapData = np.log10(np.power(10, mapData-6) + noise) + 6

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
    if bin_index_diff == -1:
        bin_index_diff = int(np.random.poisson(5))

    ratio = lumLogBinCents[0] / lumLogBinCents[0 + bin_index_diff]
    mapData = mapData * ratio

    lum_size = len(lumLogBinCents)
    for i, lum in enumerate(lumData):
        if i + bin_index_diff < lum_size:
            lumData[i] = lumData[i + bin_index_diff]
        else:
            lumData[i] = 0

    return(mapData, lumData)


