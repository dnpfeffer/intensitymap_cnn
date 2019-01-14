### import ML stuff
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

##########################################################################
### Res_NeXt #############################################################
##########################################################################
### lots of this taken from https://gist.github.com/mjdietzx/0cb95922aac14d446a6530f87b3a04ce
def get_master_res_next(modelLoc, pix_x, pix_y, numb_maps, lum_func_size,
                extra_file_name='', file_name='log_lum_6_layer_model',
                train_number=0, conv_layers=5,
                droprate=0.3, numb_layers=6, base_filters=128, threeD=True,
                luminosity_byproduct='log', kernel_size=3, cardinality=1,
                give_weights=False, loss=keras.losses.logcosh):

    def residual_network(x):
        def add_common_layers(y):
            y = layers.BatchNormalization()(y)
            y = layers.LeakyReLU()(y)

            return(y)

        def grouped_convolution(y, nb_channels, _strides):
            # when `cardinality` == 1 this is just a standard convolution
            if cardinality == 1:
                return layers.Conv3D(nb_channels, kernel_size=(3, 3, 3), strides=_strides, padding='same')(y)

            assert not nb_channels % cardinality
            _d = nb_channels // cardinality

            # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
            # and convolutions are separately performed within each group
            groups = []
            for j in range(cardinality):
                group = layers.Lambda(lambda z: z[:, :, :, j * _d:j * _d + _d])(y)
                groups.append(layers.Conv3D(_d, kernel_size=(3, 3, 3), strides=_strides, padding='same')(group))

            # the grouped convolutional layer concatenates them as the outputs of the layer
            y = layers.concatenate(groups)

            return y

        def residual_block(y, nb_channels_in, nb_channels_out, _strides=(1, 1, 1), _project_shortcut=False):
            """
            Our network consists of a stack of residual blocks. These blocks have the same topology,
            and are subject to two simple rules:
            - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
            - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.
            """
            shortcut = y

            # we modify the residual building block as a bottleneck design to make the network more economical
            y = layers.Conv3D(nb_channels_in, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(y)
            y = add_common_layers(y)

            # ResNeXt (identical to ResNet when `cardinality` == 1)
            y = grouped_convolution(y, nb_channels_in, _strides=_strides)
            y = add_common_layers(y)

            y = layers.Conv3D(nb_channels_out, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(y)
            # batch normalization is employed after aggregating the transformations and before adding to the shortcut
            y = layers.BatchNormalization()(y)

            # identity shortcuts used directly when the input and output are of the same dimensions
            if _project_shortcut or _strides != (1, 1, 1):
                # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
                # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
                shortcut = layers.Conv3D(nb_channels_out, kernel_size=(1, 1, 1), strides=_strides, padding='same')(shortcut)
                shortcut = layers.BatchNormalization()(shortcut)

            y = layers.add([shortcut, y])

            # relu is performed right after each batch normalization,
            # expect for the output of the block where relu is performed after the adding to the shortcut
            y = layers.LeakyReLU()(y)

            return y

        # conv1
        x = layers.Conv3D(64, kernel_size=(7, 7, 7), strides=(2, 2, 2), padding='same')(x)
        x = add_common_layers(x)

        # conv2
        x = layers.MaxPool3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(x)
        for i in range(3):
            project_shortcut = True if i == 0 else False
            x = residual_block(x, base_filters, base_filters*2**1, _project_shortcut=project_shortcut)

        # conv3
        for i in range(4):
            # down-sampling is performed by conv3_1, conv4_1, and conv5_1 with a stride of 2
            strides = (2, 2, 2) if i == 0 else (1, 1, 1)
            x = residual_block(x, base_filters*2**1, base_filters*2**2, _strides=strides)

        # conv4
        for i in range(6):
            strides = (2, 2, 2) if i == 0 else (1, 1, 1)
            x = residual_block(x, base_filters*2**2, base_filters*2**3, _strides=strides)

        # conv5
        for i in range(3):
           strides = (2, 2, 2) if i == 0 else (1, 1, 1)
           x = residual_block(x, base_filters*2**3, base_filters*2**4, _strides=strides)

        x = layers.GlobalAveragePooling3D()(x)
        x = layers.Dense(lum_func_size)(x)

        return(x)

    ### get the weights file name
    if give_weights:
        weight_file_name = modelLoc + file_name + extra_file_name + '_weights'
        if train_number > 0:
            weight_file_name += '_{0}'.format(int(train_number))
        weight_file_name += '.hdf5'


    image_tensor = layers.Input(shape=(pix_x, pix_y, numb_maps, 1))
    network_output = residual_network(image_tensor)
    model = models.Model(inputs=[image_tensor], outputs=[network_output])

    model.compile(loss=loss,
                  optimizer=keras.optimizers.SGD(),
                  metrics=[keras.metrics.mse])

    if give_weights:
        model.load_weights(weight_file_name)

        model.compile(loss=loss,
                       optimizer=keras.optimizers.SGD(),
                       metrics=[keras.metrics.mse])

    # print(weight_file_name)
    # master.summary()

    return(model)



##########################################################################
### master ###############################################################
##########################################################################
def get_master_(modelLoc, pix_x, pix_y, numb_maps, lum_func_size,
                extra_file_name='', file_name='full_lum_4_layer_model',
                train_number=0, give_weights=False,
                droprate=0.2, numb_layers=4, base_filters=16, threeD=False,
                luminosity_byproduct='log', kernel_size=5):

    ### get the weights file name
    weight_file_name = modelLoc + file_name + extra_file_name + '_weights'
    if train_number > 0:
        weight_file_name += '_{0}'.format(int(train_number))
    weight_file_name += '.hdf5'

    ### set which convolution to use depending on if it is 3D or not
    pool_size = 2
    if threeD:
        conv = keras.layers.Conv3D
        kernel = [kernel_size for i in range(3)]
        pool = [pool_size for i in range(3)]
        pool[-1] = 2
        kernel[-1] = 2
        strides = (1,1,1)
        input_shape = (pix_x, pix_y, numb_maps, 1)
    else:
        conv = keras.layers.Conv2D
        kernel = [kernel_size for i in range(2)]
        pool = [pool_size for i in range(2)]
        strides = (1,1)
        input_shape = (pix_x, pix_y, numb_maps)

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

    master = keras.Sequential()

    ### convolutional layer
    master.add(conv(base_filters, kernel_size=kernel, strides=strides, activation='relu', input_shape=input_shape))
    ### batch normalization
    master.add(keras.layers.BatchNormalization())
    ### use a convolution instead of a pool that acts like a pool
    master.add(conv(base_filters, kernel_size=pool, strides=pool, activation='relu'))
    ### dropout for training
    master.add(keras.layers.Dropout(droprate))

    ### loop through and add layers
    for i in range(2, numb_layers+1):
        if i == numb_layers and threeD:
            pool[-1] = 1

        ### convolutional layer
        master.add(conv(base_filters*(2**(i-1)), kernel, activation='relu'))
        ### batch normalization
        master.add(keras.layers.BatchNormalization())
        ### use a convolution instead of a pool that acts like a pool
        master.add(conv(base_filters*(2**(i-1)), kernel_size=pool, strides=pool, activation='relu'))
        ### dropout for training
        master.add(keras.layers.Dropout(droprate))

    ### flatten the network
    master.add(keras.layers.Flatten())
    ### make a dense layer for the second to last step
    master.add(keras.layers.Dense(1000, activation='relu'))
    ### finish it off with a dense layer with the number of output we want for our luminosity function
    master.add(keras.layers.Dense(lum_func_size, activation='linear'))

    master.compile(loss=loss,
                  optimizer=keras.optimizers.SGD(),
                  metrics=[keras.metrics.mse])

    if give_weights:
        master.load_weights(weight_file_name)

        master.compile(loss=loss,
                       optimizer=keras.optimizers.SGD(),
                       metrics=[keras.metrics.mse])

    # print(weight_file_name)
    # master.summary()

    return(master)

##########################################################################
### master2 ###############################################################
##########################################################################
def get_master_2(modelLoc, pix_x, pix_y, numb_maps, lum_func_size,
                extra_file_name='', file_name='full_lum_4_layer_model',
                continue_training=False, give_weights=False,
                train_number=0,
                droprate=0.2, numb_layers=4, base_filters=16, threeD=False,
                luminosity_byproduct='log', kernel_size=3):

    ### get the weights file name
    weight_file_name = modelLoc + file_name + extra_file_name + '_weights'
    if train_number > 0:
        weight_file_name += '_{0}'.format(int(train_number))
    weight_file_name += '.hdf5'

    ### set which convolution to use depending on if it is 3D or not and kernel sizes
    make_model = True
    pool_size = 2
    if threeD:
        conv = keras.layers.Conv3D
        kernel = [kernel_size for i in range(3)]
        pool = [pool_size for i in range(3)]
        strides = [1 for i in range(3)]
        input_shape = (pix_x, pix_y, numb_maps,1)
    else:
        conv = keras.layers.Conv2D
        kernel = [kernel_size for i in range(2)]
        pool = [pool_size for i in range(2)]
        strides = [1 for i in range(2)]
        input_shape = (pix_x, pix_y, numb_maps)

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

    if continue_training:
        continue_count = lnn.get_model_iteration(fileName, model_loc=modelLoc)

        master = keras.models.load_model(modelLoc + fileName + '.hdf5')
        make_model = False
        continue_name = '_{0}'.format(continue_count)
    else:
        continue_name = ''

    if make_model:
        master = keras.Sequential()

        ### convolutional layer
        master.add(conv(base_filters, kernel_size=kernel, strides=strides, activation='relu', input_shape=input_shape, padding='same'))
        ### batch normalization
        master.add(keras.layers.BatchNormalization())
        ### use a convolution instead of a pool that acts like a pool
        master.add(conv(base_filters, kernel_size=pool, strides=pool, activation='relu', padding='same'))
        ### dropout for training
        master.add(keras.layers.Dropout(droprate))

        ### loop through and add layers
        for i in range(2, numb_layers+1):
            ### convolutional layer
            master.add(conv(base_filters*(2**(i-1)), kernel, activation='relu', padding='same'))
            ### batch normalization
            master.add(keras.layers.BatchNormalization())
            ### use a convolution instead of a pool that acts like a pool
            master.add(conv(base_filters*(2**(i-1)), kernel_size=pool, strides=pool, activation='relu', padding='same'))
            ### dropout for training
            master.add(keras.layers.Dropout(droprate))

        ### flatten the network
        master.add(keras.layers.Flatten())
        ### make a dense layer for the second to last step
        master.add(keras.layers.Dense(1000, activation='relu'))
        ### finish it off with a dense layer with the number of output we want for our luminosity function
        master.add(keras.layers.Dense(lum_func_size, activation='linear'))

        master.compile(loss=loss,
                      optimizer=keras.optimizers.SGD(),
                      metrics=[keras.metrics.mse])

    if give_weights:
        master.load_weights(weight_file_name)

        master.compile(loss=loss,
                       optimizer=keras.optimizers.SGD(),
                       metrics=[keras.metrics.mse])

    # print(weight_file_name)
    # master.summary()

    return(master)

##########################################################################
### full_lum_4_layer #####################################################
##########################################################################
def get_full_lum_4_layer(modelLoc, pix_x, pix_y, numb_maps, lum_func_size,
                         extra_file_name='', file_name='full_lum_4_layer_model'):
    droprate = 0.2
    full_lum_4_layer = keras.Sequential()

    ### convolutional layer
    full_lum_4_layer.add(keras.layers.Conv3D(16, kernel_size=(5,5,5), strides=(1,1,1), activation='relu', input_shape=(pix_x, pix_y, numb_maps, 1)))
    full_lum_4_layer.add(keras.layers.BatchNormalization())
    ### use a convolution instead of a pool that acts like a pool
    full_lum_4_layer.add(keras.layers.MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2)))
    full_lum_4_layer.add(keras.layers.Dropout(droprate))

    ### convolutional layer
    full_lum_4_layer.add(keras.layers.Conv3D(32, (5,5,5), activation='relu'))
    full_lum_4_layer.add(keras.layers.BatchNormalization())
    ### use a convolution instead of a pool that acts like a pool
    full_lum_4_layer.add(keras.layers.MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2)))
    full_lum_4_layer.add(keras.layers.Dropout(droprate))

    ### convolutional layer
    full_lum_4_layer.add(keras.layers.Conv3D(64, (5,5,5), activation='relu'))
    full_lum_4_layer.add(keras.layers.BatchNormalization())
    ### use a convolution instead of a pool that acts like a pool
    full_lum_4_layer.add(keras.layers.MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2)))
    full_lum_4_layer.add(keras.layers.Dropout(droprate))

    ### convolutional layer
    full_lum_4_layer.add(keras.layers.Conv3D(128, (5,5,5), activation='relu'))
    full_lum_4_layer.add(keras.layers.BatchNormalization())
    ### use a convolution instead of a pool that acts like a pool
    full_lum_4_layer.add(keras.layers.MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2)))
    full_lum_4_layer.add(keras.layers.Dropout(droprate))

    ### flatten the network
    full_lum_4_layer.add(keras.layers.Flatten())
    ### make a dense layer for the second to last step
    full_lum_4_layer.add(keras.layers.Dense(500, activation='relu'))
    ### finish it off with a dense layer with the number of output we want for our luminosity function
    full_lum_4_layer.add(keras.layers.Dense(lum_func_size, activation='linear'))

    full_lum_4_layer.compile(loss=keras.losses.msle,
                   optimizer=keras.optimizers.SGD(),
                   metrics=[keras.metrics.mse])

    full_lum_4_layer.load_weights(modelLoc + file_name + extra_file_name + '_weights_1.hdf5')

    full_lum_4_layer.compile(loss=keras.losses.msle,
                   optimizer=keras.optimizers.SGD(),
                   metrics=[keras.metrics.mse])

    return(full_lum_4_layer)

##########################################################################
### log_lum_4_layer ######################################################
##########################################################################
def get_log_lum_4_layer(modelLoc, pix_x, pix_y, numb_maps, lum_func_size,
                        extra_file_name='', file_name='log_lum_4_layer_model'):
    droprate = 0.2
    log_lum_4_layer = keras.Sequential()

    ### convolutional layer
    log_lum_4_layer.add(keras.layers.Conv3D(16, kernel_size=(5,5,5), strides=(1,1,1), activation='relu', input_shape=(pix_x, pix_y, numb_maps, 1)))
    log_lum_4_layer.add(keras.layers.BatchNormalization())
    ### use a convolution instead of a pool that acts like a pool
    log_lum_4_layer.add(keras.layers.MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2)))
    log_lum_4_layer.add(keras.layers.Dropout(droprate))

    ### convolutional layer
    log_lum_4_layer.add(keras.layers.Conv3D(32, (5,5,5), activation='relu'))
    log_lum_4_layer.add(keras.layers.BatchNormalization())
    ### use a convolution instead of a pool that acts like a pool
    log_lum_4_layer.add(keras.layers.MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2)))
    log_lum_4_layer.add(keras.layers.Dropout(droprate))

    ### convolutional layer
    log_lum_4_layer.add(keras.layers.Conv3D(64, (5,5,5), activation='relu'))
    log_lum_4_layer.add(keras.layers.BatchNormalization())
    ### use a convolution instead of a pool that acts like a pool
    log_lum_4_layer.add(keras.layers.MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2)))
    log_lum_4_layer.add(keras.layers.Dropout(droprate))

    ### convolutional layer
    log_lum_4_layer.add(keras.layers.Conv3D(128, (5,5,5), activation='relu'))
    log_lum_4_layer.add(keras.layers.BatchNormalization())
    ### use a convolution instead of a pool that acts like a pool
    log_lum_4_layer.add(keras.layers.MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2)))
    log_lum_4_layer.add(keras.layers.Dropout(droprate))

    ### flatten the network
    log_lum_4_layer.add(keras.layers.Flatten())
    ### make a dense layer for the second to last step
    log_lum_4_layer.add(keras.layers.Dense(500, activation='relu'))
    ### finish it off with a dense layer with the number of output we want for our luminosity function
    log_lum_4_layer.add(keras.layers.Dense(lum_func_size, activation='linear'))

    log_lum_4_layer.compile(loss=keras.losses.logcosh,
                   optimizer=keras.optimizers.SGD(),
                   metrics=[keras.metrics.mse])

    log_lum_4_layer.load_weights(modelLoc + file_name + extra_file_name + '_weights.hdf5')

    log_lum_4_layer.compile(loss=keras.losses.logcosh,
                   optimizer=keras.optimizers.SGD(),
                   metrics=[keras.metrics.mse])

    return(log_lum_4_layer)

##########################################################################
### log_lum_5_layer ######################################################
##########################################################################
def get_log_lum_5_layer(modelLoc, pix_x, pix_y, numb_maps, lum_func_size,
                        extra_file_name='', file_name='log_lum_5_layer_model'):
    droprate = 0.2
    log_lum_5_layer = keras.Sequential()

    ### convolutional layer
    log_lum_5_layer.add(keras.layers.Conv3D(16, kernel_size=(5,5,5), strides=(1,1,1), activation='relu', input_shape=(pix_x, pix_y, numb_maps, 1)))
    log_lum_5_layer.add(keras.layers.BatchNormalization())
    ### use a convolution instead of a pool that acts like a pool
    log_lum_5_layer.add(keras.layers.MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2)))
    log_lum_5_layer.add(keras.layers.Dropout(droprate))

    ### convolutional layer
    log_lum_5_layer.add(keras.layers.Conv3D(32, (5,5,5), activation='relu'))
    log_lum_5_layer.add(keras.layers.BatchNormalization())
    ### use a convolution instead of a pool that acts like a pool
    log_lum_5_layer.add(keras.layers.MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2)))
    log_lum_5_layer.add(keras.layers.Dropout(droprate))

    ### convolutional layer
    log_lum_5_layer.add(keras.layers.Conv3D(64, (5,5,5), activation='relu'))
    log_lum_5_layer.add(keras.layers.BatchNormalization())
    ### use a convolution instead of a pool that acts like a pool
    log_lum_5_layer.add(keras.layers.MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2)))
    log_lum_5_layer.add(keras.layers.Dropout(droprate))

    ### convolutional layer
    log_lum_5_layer.add(keras.layers.Conv3D(128, (5,5,5), activation='relu'))
    log_lum_5_layer.add(keras.layers.BatchNormalization())
    ### use a convolution instead of a pool that acts like a pool
    log_lum_5_layer.add(keras.layers.MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2)))
    log_lum_5_layer.add(keras.layers.Dropout(droprate))

    ### convolutional layer
    log_lum_5_layer.add(keras.layers.Conv3D(256, (5,5,5), activation='relu'))
    log_lum_5_layer.add(keras.layers.BatchNormalization())
    ### use a convolution instead of a pool that acts like a pool
    log_lum_5_layer.add(keras.layers.MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2)))
    log_lum_5_layer.add(keras.layers.Dropout(droprate))

    ### flatten the network
    log_lum_5_layer.add(keras.layers.Flatten())
    ### make a dense layer for the second to last step
    log_lum_5_layer.add(keras.layers.Dense(500, activation='relu'))
    ### finish it off with a dense layer with the number of output we want for our luminosity function
    log_lum_5_layer.add(keras.layers.Dense(lum_func_size, activation='linear'))

    log_lum_5_layer.compile(loss=keras.losses.logcosh,
                   optimizer=keras.optimizers.SGD(),
                   metrics=[keras.metrics.mse])

    log_lum_5_layer.load_weights(modelLoc + file_name + extra_file_name + '_weights.hdf5')

    log_lum_5_layer.compile(loss=keras.losses.logcosh,
                   optimizer=keras.optimizers.SGD(),
                   metrics=[keras.metrics.mse])

    return(log_lum_5_layer)

##########################################################################
### log_lum_4_layer_2D ###################################################
##########################################################################
def get_log_lum_4_layer_2D(modelLoc, pix_x, pix_y, numb_maps, lum_func_size,
                           extra_file_name='', file_name='log_lum_4_layer_2D_model',
                          filter_ratio=1):
    droprate = 0.2
    log_lum_4_layer_2D = keras.Sequential()

    ### convolutional layer
    log_lum_4_layer_2D.add(keras.layers.Conv2D(16*filter_ratio, kernel_size=(5,5), strides=(1,1), activation='relu', input_shape=(pix_x, pix_y, numb_maps)))
    log_lum_4_layer_2D.add(keras.layers.BatchNormalization())
    ### use a convolution instead of a pool that acts like a pool
    log_lum_4_layer_2D.add(keras.layers.Conv2D(16*filter_ratio, kernel_size=(2,2), strides=(2,2), activation='relu'))
    # model2.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    log_lum_4_layer_2D.add(keras.layers.Dropout(droprate))

    ### convolutional layer
    log_lum_4_layer_2D.add(keras.layers.Conv2D(32*filter_ratio, (5,5), activation='relu'))
    log_lum_4_layer_2D.add(keras.layers.BatchNormalization())
    ### use a convolution instead of a pool that acts like a pool
    log_lum_4_layer_2D.add(keras.layers.Conv2D(32*filter_ratio, kernel_size=(2,2), strides=(2,2), activation='relu'))
    # model2.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    log_lum_4_layer_2D.add(keras.layers.Dropout(droprate))

    ### convolutional layer
    log_lum_4_layer_2D.add(keras.layers.Conv2D(64*filter_ratio, (5,5), activation='relu'))
    log_lum_4_layer_2D.add(keras.layers.BatchNormalization())
    ### use a convolution instead of a pool that acts like a pool
    log_lum_4_layer_2D.add(keras.layers.Conv2D(64*filter_ratio, kernel_size=(2,2), strides=(2,2), activation='relu'))
    # model2.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    log_lum_4_layer_2D.add(keras.layers.Dropout(droprate))

    ### convolutional layer
    log_lum_4_layer_2D.add(keras.layers.Conv2D(128*filter_ratio, (5,5), activation='relu'))
    log_lum_4_layer_2D.add(keras.layers.BatchNormalization())
    ### use a convolution instead of a pool that acts like a pool
    log_lum_4_layer_2D.add(keras.layers.Conv2D(128*filter_ratio, kernel_size=(2,2), strides=(2,2), activation='relu'))
    # model2.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    log_lum_4_layer_2D.add(keras.layers.Dropout(droprate))

    ### flatten the network
    log_lum_4_layer_2D.add(keras.layers.Flatten())
    ### make a dense layer for the second to last step
    log_lum_4_layer_2D.add(keras.layers.Dense(1000, activation='relu'))
    log_lum_4_layer_2D.add(keras.layers.Dense(lum_func_size, activation='linear'))

    log_lum_4_layer_2D.compile(loss=keras.losses.logcosh,
                  optimizer=keras.optimizers.SGD(),
                  metrics=[keras.metrics.mse])

    log_lum_4_layer_2D.load_weights(modelLoc + file_name + extra_file_name + '_weights.hdf5')

    log_lum_4_layer_2D.compile(loss=keras.losses.logcosh,
                  optimizer=keras.optimizers.SGD(),
                  metrics=[keras.metrics.mse])

    return(log_lum_4_layer_2D)

##########################################################################
### fullL_lum_4_layer_2D ###################################################
##########################################################################
def get_fullL_lum_4_layer_2D(modelLoc, pix_x, pix_y, numb_maps, lum_func_size,
                           extra_file_name='', file_name='log_lum_4_layer_2D_model',
                           filter_ratio=1):
    droprate = 0.2
    fullL_lum_4_layer_2D = keras.Sequential()

    ### convolutional layer
    fullL_lum_4_layer_2D.add(keras.layers.Conv2D(16*filter_ratio, kernel_size=(5,5), strides=(1,1), activation='relu', input_shape=(pix_x, pix_y, numb_maps)))
    fullL_lum_4_layer_2D.add(keras.layers.BatchNormalization())
    ### use a convolution instead of a pool that acts like a pool
    fullL_lum_4_layer_2D.add(keras.layers.Conv2D(16*filter_ratio, kernel_size=(2,2), strides=(2,2), activation='relu'))
    # model2.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    fullL_lum_4_layer_2D.add(keras.layers.Dropout(droprate))

    ### convolutional layer
    fullL_lum_4_layer_2D.add(keras.layers.Conv2D(32*filter_ratio, (5,5), activation='relu'))
    fullL_lum_4_layer_2D.add(keras.layers.BatchNormalization())
    ### use a convolution instead of a pool that acts like a pool
    fullL_lum_4_layer_2D.add(keras.layers.Conv2D(32*filter_ratio, kernel_size=(2,2), strides=(2,2), activation='relu'))
    # model2.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    fullL_lum_4_layer_2D.add(keras.layers.Dropout(droprate))

    ### convolutional layer
    fullL_lum_4_layer_2D.add(keras.layers.Conv2D(64*filter_ratio, (5,5), activation='relu'))
    fullL_lum_4_layer_2D.add(keras.layers.BatchNormalization())
    ### use a convolution instead of a pool that acts like a pool
    fullL_lum_4_layer_2D.add(keras.layers.Conv2D(64*filter_ratio, kernel_size=(2,2), strides=(2,2), activation='relu'))
    # model2.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    fullL_lum_4_layer_2D.add(keras.layers.Dropout(droprate))

    ### convolutional layer
    fullL_lum_4_layer_2D.add(keras.layers.Conv2D(128*filter_ratio, (5,5), activation='relu'))
    fullL_lum_4_layer_2D.add(keras.layers.BatchNormalization())
    ### use a convolution instead of a pool that acts like a pool
    fullL_lum_4_layer_2D.add(keras.layers.Conv2D(128*filter_ratio, kernel_size=(2,2), strides=(2,2), activation='relu'))
    # model2.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    fullL_lum_4_layer_2D.add(keras.layers.Dropout(droprate))

    ### flatten the network
    fullL_lum_4_layer_2D.add(keras.layers.Flatten())
    ### make a dense layer for the second to last step
    fullL_lum_4_layer_2D.add(keras.layers.Dense(1000, activation='relu'))
    fullL_lum_4_layer_2D.add(keras.layers.Dense(lum_func_size, activation='linear'))

    fullL_lum_4_layer_2D.compile(loss=keras.losses.logcosh,
                  optimizer=keras.optimizers.SGD(),
                  metrics=[keras.metrics.mse])

    fullL_lum_4_layer_2D.load_weights(modelLoc + file_name + extra_file_name + '_weights.hdf5')

    fullL_lum_4_layer_2D.compile(loss=keras.losses.mse,
                  optimizer=keras.optimizers.SGD(),
                  metrics=[keras.metrics.mse])

    return(fullL_lum_4_layer_2D)

##########################################################################
### log_lum_5_layer_2D ###################################################
##########################################################################
def get_log_lum_5_layer_2D(modelLoc, pix_x, pix_y, numb_maps, lum_func_size,
                           extra_file_name='', file_name='log_lum_5_layer_2D_model',
                           filter_ratio=1):
    droprate = 0.2
    log_lum_5_layer_2D = keras.Sequential()

    ### convolutional layer
    log_lum_5_layer_2D.add(keras.layers.Conv2D(16*filter_ratio, kernel_size=(5,5), strides=(1,1), activation='relu', input_shape=(pix_x, pix_y, numb_maps)))
    log_lum_5_layer_2D.add(keras.layers.BatchNormalization())
    ### use a convolution instead of a pool that acts like a pool
    log_lum_5_layer_2D.add(keras.layers.Conv2D(16*filter_ratio, kernel_size=(2,2), strides=(2,2), activation='relu'))
    # model2.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    log_lum_5_layer_2D.add(keras.layers.Dropout(droprate))

    ### convolutional layer
    log_lum_5_layer_2D.add(keras.layers.Conv2D(32*filter_ratio, (5,5), activation='relu'))
    log_lum_5_layer_2D.add(keras.layers.BatchNormalization())
    ### use a convolution instead of a pool that acts like a pool
    log_lum_5_layer_2D.add(keras.layers.Conv2D(32*filter_ratio, kernel_size=(2,2), strides=(2,2), activation='relu'))
    # model2.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    log_lum_5_layer_2D.add(keras.layers.Dropout(droprate))

    ### convolutional layer
    log_lum_5_layer_2D.add(keras.layers.Conv2D(64*filter_ratio, (5,5), activation='relu'))
    log_lum_5_layer_2D.add(keras.layers.BatchNormalization())
    ### use a convolution instead of a pool that acts like a pool
    log_lum_5_layer_2D.add(keras.layers.Conv2D(64*filter_ratio, kernel_size=(2,2), strides=(2,2), activation='relu'))
    # model2.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    log_lum_5_layer_2D.add(keras.layers.Dropout(droprate))

    ### convolutional layer
    log_lum_5_layer_2D.add(keras.layers.Conv2D(128*filter_ratio, (5,5), activation='relu'))
    log_lum_5_layer_2D.add(keras.layers.BatchNormalization())
    ### use a convolution instead of a pool that acts like a pool
    log_lum_5_layer_2D.add(keras.layers.Conv2D(128*filter_ratio, kernel_size=(2,2), strides=(2,2), activation='relu'))
    # model2.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    log_lum_5_layer_2D.add(keras.layers.Dropout(droprate))

    ### convolutional layer
    log_lum_5_layer_2D.add(keras.layers.Conv2D(256*filter_ratio, (5,5), activation='relu'))
    log_lum_5_layer_2D.add(keras.layers.BatchNormalization())
    ### use a convolution instead of a pool that acts like a pool
    log_lum_5_layer_2D.add(keras.layers.Conv2D(256*filter_ratio, kernel_size=(2,2), strides=(2,2), activation='relu'))
    # model2.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    log_lum_5_layer_2D.add(keras.layers.Dropout(droprate))

    ### flatten the network
    log_lum_5_layer_2D.add(keras.layers.Flatten())
    ### make a dense layer for the second to last step
    log_lum_5_layer_2D.add(keras.layers.Dense(1000, activation='relu'))
    log_lum_5_layer_2D.add(keras.layers.Dense(lum_func_size, activation='linear'))

    log_lum_5_layer_2D.compile(loss=keras.losses.logcosh,
                  optimizer=keras.optimizers.SGD(),
                  metrics=[keras.metrics.mse])

    log_lum_5_layer_2D.load_weights(modelLoc + file_name + extra_file_name + '_weights.hdf5')

    log_lum_5_layer_2D.compile(loss=keras.losses.logcosh,
                  optimizer=keras.optimizers.SGD(),
                  metrics=[keras.metrics.mse])

    return(log_lum_5_layer_2D)