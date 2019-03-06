# import ML stuff
# import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

##########################################################################
### Res_NeXt #############################################################
##########################################################################
### lots of this taken from https://gist.github.com/mjdietzx/0cb95922aac14d446a6530f87b3a04ce
def get_master_res_next(modelLoc, pix_x, pix_y, numb_maps, lum_func_size,
                extra_file_name='', file_name='log_lum_6_layer_model',
                train_number=0, droprate=0.3, base_filters=128, cardinality=1,
                batchNorm_momentum=0.99, dense_layer=1000,
                give_weights=False, loss=keras.losses.logcosh, use_bias=True):

    def residual_network(x):
        def add_common_layers(y):
            y = layers.BatchNormalization(momentum=batchNorm_momentum)(y)
            y = layers.LeakyReLU()(y)

            return(y)

        def grouped_convolution(y, nb_channels, _strides, use_bias=False):
            # when `cardinality` == 1 this is just a standard convolution
            if cardinality == 1:
                return layers.Conv3D(nb_channels, kernel_size=(3, 3, 3), strides=_strides,
                    padding='same', use_bias=use_bias)(y)

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

        def residual_block(y, nb_channels_in, nb_channels_out, _strides=(1, 1, 1), _project_shortcut=False, use_bias=False):
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
            y = grouped_convolution(y, nb_channels_in, _strides=_strides, use_bias=use_bias)
            y = add_common_layers(y)

            y = layers.Conv3D(nb_channels_out, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(y)
            # batch normalization is employed after aggregating the transformations and before adding to the shortcut
            y = layers.BatchNormalization(momentum=batchNorm_momentum)(y)

            # identity shortcuts used directly when the input and output are of the same dimensions
            if _project_shortcut or _strides != (1, 1, 1):
                # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
                # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
                shortcut = layers.Conv3D(nb_channels_out, kernel_size=(1, 1, 1), strides=_strides, padding='same')(shortcut)
                shortcut = layers.BatchNormalization(momentum=batchNorm_momentum)(shortcut)

            y = layers.add([shortcut, y])

            # relu is performed right after each batch normalization,
            # expect for the output of the block where relu is performed after the adding to the shortcut
            y = layers.LeakyReLU()(y)

            return y

        # conv1
        x = layers.Conv3D(base_filters, kernel_size=(7, 7, 7), strides=(2, 2, 2),
            padding='same', use_bias=use_bias)(x)
        x = add_common_layers(x)

        # conv2
        x = layers.MaxPool3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(x)
        for i in range(3):
            project_shortcut = True if i == 0 else False
            x = residual_block(x, base_filters*2**1, base_filters*2**2,
                _project_shortcut=project_shortcut,
                use_bias=use_bias)

        # conv3
        for i in range(4):
            # down-sampling is performed by conv3_1, conv4_1, and conv5_1 with a stride of 2
            strides = (2, 2, 2) if i == 0 else (1, 1, 1)
            x = residual_block(x, base_filters*2**2, base_filters*2**3,
                _strides=strides,
                use_bias=use_bias)

        # conv4
        for i in range(6):
            strides = (2, 2, 2) if i == 0 else (1, 1, 1)
            x = residual_block(x, base_filters*2**3, base_filters*2**4,
                _strides=strides,
                use_bias=use_bias)

        # conv5
        for i in range(3):
           strides = (2, 2, 2) if i == 0 else (1, 1, 1)
           x = residual_block(x, base_filters*2**4, base_filters*2**5,
            _strides=strides,
            use_bias=use_bias)

        x = layers.GlobalAveragePooling3D()(x)
        x = layers.Dense(dense_layer)(x)
        x = layers.BatchNormalization(momentum=batchNorm_momentum)(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(droprate)(x)
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
                         optimizer=keras.optimizers.Adam(),
                         metrics=[keras.metrics.mse])

    if give_weights:
        model.load_weights(weight_file_name)

        model.compile(loss=loss,
                         optimizer=keras.optimizers.Adam(),
                         metrics=[keras.metrics.mse])

    # print(weight_file_name)
    # master.summary()

    return(model)

##########################################################################
### Res_NeXt2 #############################################################
##########################################################################
### lots of this taken from https://gist.github.com/mjdietzx/0cb95922aac14d446a6530f87b3a04ce
def get_master_res_next2(model_params, extra_file_name='', give_weights=False, cardinality=1):
    batchNorm_momentum = model_params.batchNorm_momentum

    # if hasattr(model_params, 'file_name'):
    try:
        file_name = model_params.file_name
    # else:
    except AttributeError:
        file_name = ''

    model = get_master_res_next(model_params.modelLoc, model_params.pix_x,
                model_params.pix_y, model_params.numb_maps, model_params.lum_func_size,
                extra_file_name=extra_file_name,
                file_name=file_name ,
                train_number=model_params.train_number,
                droprate=model_params.droprate,
                base_filters=model_params.base_filters,
                cardinality=cardinality,
                batchNorm_momentum=model_params.batchNorm_momentum,
                dense_layer=model_params.dense_layer,
                give_weights=give_weights,
                loss=model_params.loss,
                use_bias=model_params.use_bias)

    # def residual_network(x):
    #     def add_common_layers(y):
    #         y = layers.BatchNormalization(momentum=batchNorm_momentum)(y)
    #         y = layers.LeakyReLU()(y)

    #         return(y)

    #     def grouped_convolution(y, nb_channels, _strides, use_bias=False):
    #         # when `cardinality` == 1 this is just a standard convolution
    #         if cardinality == 1:
    #             return layers.Conv3D(nb_channels, kernel_size=(3, 3, 3), strides=_strides,
    #                 padding='same', use_bias=use_bias)(y)

    #         assert not nb_channels % cardinality
    #         _d = nb_channels // cardinality

    #         # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
    #         # and convolutions are separately performed within each group
    #         groups = []
    #         for j in range(cardinality):
    #             group = layers.Lambda(lambda z: z[:, :, :, j * _d:j * _d + _d])(y)
    #             groups.append(layers.Conv3D(_d, kernel_size=(3, 3, 3), strides=_strides, padding='same')(group))

    #         # the grouped convolutional layer concatenates them as the outputs of the layer
    #         y = layers.concatenate(groups)

    #         return y

    #     def residual_block(y, nb_channels_in, nb_channels_out, _strides=(1, 1, 1), _project_shortcut=False, use_bias=False):
    #         """
    #         Our network consists of a stack of residual blocks. These blocks have the same topology,
    #         and are subject to two simple rules:
    #         - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
    #         - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.
    #         """
    #         shortcut = y

    #         # we modify the residual building block as a bottleneck design to make the network more economical
    #         y = layers.Conv3D(nb_channels_in, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(y)
    #         y = add_common_layers(y)

    #         # ResNeXt (identical to ResNet when `cardinality` == 1)
    #         y = grouped_convolution(y, nb_channels_in, _strides=_strides, use_bias=use_bias)
    #         y = add_common_layers(y)

    #         y = layers.Conv3D(nb_channels_out, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(y)
    #         # batch normalization is employed after aggregating the transformations and before adding to the shortcut
    #         y = layers.BatchNormalization(momentum=batchNorm_momentum)(y)

    #         # identity shortcuts used directly when the input and output are of the same dimensions
    #         if _project_shortcut or _strides != (1, 1, 1):
    #             # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
    #             # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
    #             shortcut = layers.Conv3D(nb_channels_out, kernel_size=(1, 1, 1), strides=_strides, padding='same')(shortcut)
    #             shortcut = layers.BatchNormalization(momentum=batchNorm_momentum)(shortcut)

    #         y = layers.add([shortcut, y])

    #         # relu is performed right after each batch normalization,
    #         # expect for the output of the block where relu is performed after the adding to the shortcut
    #         y = layers.LeakyReLU()(y)

    #         return y

    #     # conv1
    #     x = layers.Conv3D(model_params.base_filters, kernel_size=(7, 7, 7), strides=(2, 2, 2),
    #         padding='same', use_bias=model_params.use_bias)(x)
    #     x = add_common_layers(x)

    #     # conv2
    #     x = layers.MaxPool3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(x)
    #     for i in range(3):
    #         project_shortcut = True if i == 0 else False
    #         x = residual_block(x, model_params.base_filters*2**1, model_params.base_filters*2**1,
    #             _project_shortcut=project_shortcut,
    #             use_bias=model_params.use_bias)

    #     # conv3
    #     for i in range(4):
    #         # down-sampling is performed by conv3_1, conv4_1, and conv5_1 with a stride of 2
    #         strides = (2, 2, 2) if i == 0 else (1, 1, 1)
    #         x = residual_block(x, model_params.base_filters*2**2, model_params.base_filters*2**2,
    #             _strides=strides,
    #             use_bias=model_params.use_bias)

    #     # conv4
    #     for i in range(6):
    #         strides = (2, 2, 2) if i == 0 else (1, 1, 1)
    #         x = residual_block(x, model_params.base_filters*2**3, model_params.base_filters*2**3,
    #             _strides=strides,
    #             use_bias=model_params.use_bias)

    #     # conv5
    #     for i in range(3):
    #        strides = (2, 2, 2) if i == 0 else (1, 1, 1)
    #        x = residual_block(x, model_params.base_filters*2**4, model_params.base_filters*2**4,
    #         _strides=strides,
    #         use_bias=model_params.use_bias)

    #     x = layers.GlobalAveragePooling3D()(x)
    #     x = layers.Dense(model_params.dense_layer)(x)
    #     x = layers.BatchNormalization(momentum=batchNorm_momentum)(x)
    #     x = layers.ReLU()(x)
    #     x = layers.Dropout(model_params.droprate)(x)
    #     x = layers.Dense(model_params.lum_func_size)(x)

    #     return(x)

    # ### get the weights file name
    # if give_weights:
    #     weight_file_name = model_params.modelLoc + model_params.file_name + extra_file_name + '_weights'
    #     if model_params.train_number > 0:
    #         weight_file_name += '_{0}'.format(int(model_params.train_number))
    #     weight_file_name += '.hdf5'


    # image_tensor = layers.Input(shape=(model_params.pix_x, model_params.pix_y, model_params.numb_maps, 1))
    # network_output = residual_network(image_tensor)
    # model = models.Model(inputs=[image_tensor], outputs=[network_output])


    # model.compile(loss=model_params.loss,
    #                      optimizer=keras.optimizers.Adam(),
    #                      metrics=[keras.metrics.mse])

    # if give_weights:
    #     model.load_weights(weight_file_name)

    #     model.compile(loss=model_params.loss,
    #                      optimizer=keras.optimizers.Adam(),
    #                      metrics=[keras.metrics.mse])

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
                give_weights=False,
                train_number=0,
                droprate=0.2, numb_layers=4, base_filters=16, threeD=False,
                luminosity_byproduct='log', kernel_size=3,
                dense_layer=1000, use_bias=True):

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
        kernel[-1] = 3
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

    if make_model:
        master = keras.Sequential()

        ### convolutional layer
        master.add(conv(base_filters, kernel_size=kernel, strides=strides, activation='relu', input_shape=input_shape,
            padding='same', use_bias=False))
        ### batch normalization
        master.add(keras.layers.BatchNormalization())
        ### dropout for training
        master.add(keras.layers.Dropout(droprate))
        ### use a convolution instead of a pool that acts like a pool
        master.add(conv(base_filters, kernel_size=pool, strides=pool, activation='relu', padding='same', use_bias=False))

        ### convolutional layer
        master.add(conv(base_filters*2, kernel, activation='relu', padding='same', use_bias=False))
        ### batch normalization
        master.add(keras.layers.BatchNormalization())
        ### dropout for training
        master.add(keras.layers.Dropout(droprate))
        ### use a convolution instead of a pool that acts like a pool
        master.add(conv(base_filters*2, kernel_size=pool, strides=pool, activation='relu', padding='same', use_bias=False))

        ### loop through and add layers
        for i in range(3, numb_layers+1):
            ### convolutional layer
            master.add(conv(base_filters*(2**(i-1)), kernel, activation='relu', padding='same', use_bias=use_bias))
            ### batch normalization
            master.add(keras.layers.BatchNormalization())
            ### dropout for training
            master.add(keras.layers.Dropout(droprate))
            ### use a convolution instead of a pool that acts like a pool
            master.add(conv(base_filters*(2**(i-1)), kernel_size=pool, strides=pool, activation='relu', padding='same', use_bias=use_bias))

        ### flatten the network
        master.add(keras.layers.Flatten())
        ### make a dense layer for the second to last step
        master.add(keras.layers.Dense(dense_layer, activation='relu', use_bias=use_bias))
        ### dropout for training
        master.add(keras.layers.Dropout(droprate))
        ### finish it off with a dense layer with the number of output we want for our luminosity function
        master.add(keras.layers.Dense(lum_func_size, activation='linear'))

        lr = 0.01
        momentum = 0.7
        decay_rate = lr/100

        master.compile(loss=loss,
                    optimizer=keras.optimizers.SGD(lr=lr, momentum=momentum, decay=decay_rate),
                    metrics=[keras.metrics.mse])

    if give_weights:
        master.load_weights(weight_file_name)

        master.compile(loss=loss,
                       optimizer=keras.optimizers.SGD(lr=lr, momentum=momentum, decay=decay_rate),
                       metrics=[keras.metrics.mse])

    # print(weight_file_name)
    # master.summary()

    return(master)

##########################################################################
### master3 ###############################################################
##########################################################################
def get_master_3(model_params, extra_file_name='',
                    give_weights=False):
                # modelLoc, pix_x, pix_y, numb_maps, lum_func_size,
                # extra_file_name='', file_name='full_lum_4_layer_model',
                # give_weights=False,
                # train_number=0,
                # droprate=0.2, numb_layers=4, base_filters=16, threeD=False,
                # luminosity_byproduct='log', kernel_size=3,
                # dense_layer=1000, use_bias=True):

    lr = 0.01
    momentum = 0.7
    decay_rate = lr/100

    if model_params.make_model:
        ### set which convolution to use depending on if it is 3D or not and kernel sizes
        pool_size = 2
        if model_params.ThreeD:
            conv = keras.layers.Conv3D
            kernel = [model_params.kernel_size for i in range(3)]
            kernel[-1] = 3
            pool = [pool_size for i in range(3)]
            strides = [1 for i in range(3)]
            input_shape = (model_params.pix_x, model_params.pix_y, model_params.numb_maps,1)
        else:
            conv = keras.layers.Conv2D
            kernel = [model_params.kernel_size for i in range(2)]
            pool = [pool_size for i in range(2)]
            strides = [1 for i in range(2)]
            input_shape = (model_params.pix_x, model_params.pix_y, model_params.numb_maps)

        master = keras.Sequential()

        ### convolutional layer
        master.add(conv(model_params.base_filters, kernel_size=kernel, strides=strides, activation='relu', input_shape=input_shape,
            padding='same', use_bias=False))
        ### batch normalization
        master.add(keras.layers.BatchNormalization())
        ### dropout for training
        master.add(keras.layers.Dropout(model_params.droprate))
        ### use a convolution instead of a pool that acts like a pool
        master.add(conv(model_params.base_filters, kernel_size=pool, strides=pool, activation='relu', padding='same', use_bias=False))

        ### convolutional layer
        master.add(conv(model_params.base_filters*2, kernel, activation='relu', padding='same', use_bias=False))
        ### batch normalization
        master.add(keras.layers.BatchNormalization())
        ### dropout for training
        master.add(keras.layers.Dropout(model_params.droprate))
        ### use a convolution instead of a pool that acts like a pool
        master.add(conv(model_params.base_filters*2, kernel_size=pool, strides=pool, activation='relu', padding='same', use_bias=False))

        ### loop through and add layers
        for i in range(3, model_params.numb_layers+1):
            ### convolutional layer
            master.add(conv(model_params.base_filters*(2**(i-1)), kernel, activation='relu', padding='same', use_bias=model_params.use_bias))
            ### batch normalization
            master.add(keras.layers.BatchNormalization())
            ### dropout for training
            master.add(keras.layers.Dropout(model_params.droprate))
            ### use a convolution instead of a pool that acts like a pool
            master.add(conv(model_params.base_filters*(2**(i-1)), kernel_size=pool, strides=pool, activation='relu', padding='same', use_bias=model_params.use_bias))

        ### flatten the network
        master.add(keras.layers.Flatten())
        ### make a dense layer for the second to last step
        master.add(keras.layers.Dense(model_params.dense_layer, activation='relu', use_bias=model_params.use_bias))
        ### dropout for training
        master.add(keras.layers.Dropout(model_params.droprate))
        ### finish it off with a dense layer with the number of output we want for our luminosity function
        master.add(keras.layers.Dense(model_params.lum_func_size, activation='linear'))

        master.compile(loss=model_params.loss,
                    optimizer=keras.optimizers.SGD(lr=lr, momentum=momentum, decay=decay_rate),
                    metrics=[keras.metrics.mse])

    if give_weights:
        ### get the weights file name
        weight_file_name = model_params.modelLoc + model_params.fileName + extra_file_name + '_weights'
        if model_params.train_number > 0:
            weight_file_name += '_{0}'.format(int(model_params.train_number))
        weight_file_name += '.hdf5'

        master.load_weights(weight_file_name)

        master.compile(loss=model_params.loss,
                       optimizer=keras.optimizers.SGD(lr=lr, momentum=momentum, decay=decay_rate),
                       metrics=[keras.metrics.mse])

    # print(weight_file_name)
    # master.summary()

    return(master)

##########################################################################
### ANN ##################################################################
##########################################################################
def get_master_ann(modelLoc, pix_x, pix_y, numb_maps, lum_func_size,
                extra_file_name='', file_name='full_lum_4_layer_model',
                give_weights=False,
                train_number=0,
                droprate=0.2, numb_layers=4, base_filters=16,
                luminosity_byproduct='log', use_bias=True):

    ### get the weights file name
    weight_file_name = modelLoc + file_name + extra_file_name + '_weights'
    if train_number > 0:
        weight_file_name += '_{0}'.format(int(train_number))
    weight_file_name += '.hdf5'

    ### set which convolution to use depending on if it is 3D or not and kernel sizes
    make_model = True
    # pool_size = 2
    # if threeD:
    #     conv = keras.layers.Conv3D
    #     kernel = [kernel_size for i in range(3)]
    #     pool = [pool_size for i in range(3)]
    #     strides = [1 for i in range(3)]
    #     input_shape = (pix_x, pix_y, numb_maps,1)
    # else:
    #     conv = keras.layers.Conv2D
    #     kernel = [kernel_size for i in range(2)]
    #     pool = [pool_size for i in range(2)]
    #     strides = [1 for i in range(2)]
    #     input_shape = (pix_x, pix_y, numb_maps)

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

    image_tensor = layers.Input(shape=(pix_x, pix_y, numb_maps, 1))

    x = layers.Flatten()(image_tensor)

    x = layers.Dense(1000, activation='relu')(x)
    x = layers.Dropout(droprate)(x)

    x = layers.Dense(1000, activation='relu')(x)
    x = layers.Dropout(droprate)(x)

    x = layers.Dense(500, activation='relu')(x)
    x = layers.Dropout(droprate)(x)

    x = layers.Dense(100, activation='relu')(x)
    x = layers.Dropout(droprate)(x)

    x = layers.Dense(lum_func_size, activation='linear')(x)
    network_output = layers.Dropout(droprate)(x)

    master = models.Model(inputs=[image_tensor], outputs=[network_output])

    lr = 0.001
    momentum = 0.7
    decay_rate = lr/100

    master.compile(loss=loss,
                optimizer=keras.optimizers.SGD(lr=lr, momentum=momentum, decay=decay_rate),
                metrics=[keras.metrics.mse])

    if give_weights:
        master.load_weights(weight_file_name)

        master.compile(loss=loss,
                       optimizer=keras.optimizers.SGD(lr=lr, momentum=momentum, decay=decay_rate),
                       metrics=[keras.metrics.mse])

    master.summary()

    return(master)

##########################################################################
### adam #################################################################
##########################################################################
def get_master_adam(modelLoc, pix_x, pix_y, numb_maps, lum_func_size,
                extra_file_name='', file_name='full_lum_4_layer_model',
                give_weights=False,
                train_number=0,
                droprate=0.2, numb_layers=4, base_filters=16, threeD=False,
                luminosity_byproduct='log', kernel_size=3,
                dense_layer=1000, use_bias=True):

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
        kernel[-1] = 3
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

    if make_model:
        master = keras.Sequential()

        ### convolutional layer
        master.add(conv(base_filters, kernel_size=kernel, strides=strides, activation='relu', input_shape=input_shape,
            padding='same', use_bias=use_bias))
        ### batch normalization
        master.add(keras.layers.BatchNormalization())
        ### dropout for training
        master.add(keras.layers.Dropout(droprate))
        ### use a convolution instead of a pool that acts like a pool
        master.add(conv(base_filters, kernel_size=pool, strides=pool, activation='relu', padding='same', use_bias=use_bias))

        ### convolutional layer
        master.add(conv(base_filters*2, kernel, activation='relu', padding='same', use_bias=use_bias))
        ### batch normalization
        master.add(keras.layers.BatchNormalization())
        ### dropout for training
        master.add(keras.layers.Dropout(droprate))
        ### use a convolution instead of a pool that acts like a pool
        master.add(conv(base_filters*2, kernel_size=pool, strides=pool, activation='relu', padding='same', use_bias=use_bias))

        ### loop through and add layers
        for i in range(3, numb_layers+1):
            ### convolutional layer
            master.add(conv(base_filters*(2**(i-1)), kernel, activation='relu', padding='same', use_bias=True))
            ### batch normalization
            master.add(keras.layers.BatchNormalization())
            ### dropout for training
            master.add(keras.layers.Dropout(droprate))
            ### use a convolution instead of a pool that acts like a pool
            master.add(conv(base_filters*(2**(i-1)), kernel_size=pool, strides=pool, activation='relu', padding='same', use_bias=True))

        ### flatten the network
        master.add(keras.layers.Flatten())
        ### make a dense layer for the second to last step
        master.add(keras.layers.Dense(dense_layer, activation='relu', use_bias=True))
        ### dropout for training
        master.add(keras.layers.Dropout(droprate))
        ### finish it off with a dense layer with the number of output we want for our luminosity function
        master.add(keras.layers.Dense(lum_func_size, activation='linear'))

        master.compile(loss=loss,
                    optimizer=keras.optimizers.Adam(),
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
### Categorizer ##########################################################
##########################################################################
def get_master_categorizer(modelLoc, pix_x, pix_y, numb_maps, base_model=get_master_2,
                extra_file_name='', file_name='full_lum_4_layer_model',
                train_number=0, base_file_name='',
                give_weights=False, use_base_weights=False, **kwargs):

    if 'threeD' in kwargs:
        threeD = kwargs['threeD']
    else:
        threeD = False

    master = base_model(modelLoc, pix_x, pix_y, numb_maps, 1,
        extra_file_name=extra_file_name, file_name=base_file_name,
        give_weights=use_base_weights, train_number=train_number,
        **kwargs)

    master.layers.pop()
    master.layers.pop()

    x = master.layers[-1].output
    if threeD:
        x = layers.GlobalAveragePooling3D()(x)
    else:
        x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(500, activation='relu')(x)
    o = layers.Dense(3, activation='relu')(x)

    master_categorizer = models.Model(inputs=master.layers[0].input, outpus=o)

    master_categorizer.compile(loss=keras.losses.categorical_crossentropy,
                   optimizer=keras.optimizers.Adam(),
                   metrics=[keras.metrics.categorical_accuracy])

    if give_weights:
        ### get the weights file name
        weight_file_name = modelLoc + file_name + extra_file_name + '_weights'
        if train_number > 0:
            weight_file_name += '_{0}'.format(int(train_number))
        weight_file_name += '.hdf5'

        master_categorizer.load_weights(weight_file_name)

        master_categorizer.compile(loss=keras.losses.categorical_crossentropy,
                       optimizer=keras.optimizers.Adam(),
                       metrics=[keras.metrics.categorical_accuracy])

    return(master_categorizer)


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
