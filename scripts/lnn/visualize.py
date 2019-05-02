import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter

from .ioFuncs import *
from .preprocessing import *

import tensorflow as tf
from tensorflow import keras as k

##########################################
### Jupyter Notebook Plotting Tools ###
##########################################

# plot_single_history
def plot_single_history(history, start_loc=1, do_val=True):
    fig, ax = plt.subplots(figsize=(9, 6))
    # fig, ax = plt.subplots()

    key = 'loss'
    # color = 'k'
    p = ax.semilogy(range(len(history[key]))[start_loc:], history[key][start_loc:],
    label='Training')
    if do_val:
        color = p[-1].get_color()
        ax.semilogy(range(len(history['val_' + key]))[start_loc:],
        history['val_' + key][start_loc:], ls='--', label='Validation', color=color)

    ### display the plot
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    fig.tight_layout()
    return(ax)

# plots the history of training a model and compares two metrics at the same time
def history_compare_two_metrics(history, metrics=['loss', 'mean_squared_error'], start_loc=1, do_val=True):
    # set up plot
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax_list = [ax1, ax2]
    colors = ['r', 'b']

    # don't allow more then 2 metrics to be given
    if len(metrics) > 2:
        print("More then 2 metrics were given.  Please only give 2")
        return()

    # do the plot for each metric given
    for i in range(len(metrics)):
        key = metrics[i]
        ax = ax_list[i]
        color = colors[i]

        # actual plotting on each axis
        ax.semilogy(range(len(history[key]))[start_loc:], history[key][start_loc:], label=key, color=color)
        if do_val:
            ax.semilogy(range(len(history['val_' + key]))[start_loc:], history['val_' + key][start_loc:], color=color, ls='--')
        ax.set_ylabel(key, color=color)
        ax.tick_params('y', which='both', labelcolor=color)

    # set up the legend to show training and validation line types
    if do_val:
        custom_lines = [Line2D([0], [0], linestyle='-', color='k', lw=1),
                    Line2D([0], [0], linestyle='--', color='k', lw=1)]
        ax1.legend(custom_lines, ['Training', 'Validation'])
    else:
        custom_lines = [Line2D([0], [0], linestyle='-', color='k', lw=1)]
        ax1.legend(custom_lines, ['Training'])

    # display the plot
    ax1.set_xlabel('Epoch')
    fig.tight_layout()
    plt.show()

    return()

# get the predicted and simulated luminosity function byproduct for a given model and map
def test_model(model, base, base_number, luminosity_byproduct='log', threeD=False, evaluate=True,
    log_input=False, make_map_noisy=0, pre_pool=1, pre_pool_z=25, lum_func_size=None,
    add_foregrounds=False, random_foreground_params=False, rotate=0,
    gaussian_smoothing=0):

    # get the simulated map and luminosity byproduct
    cur_map, cur_lum = fileToMapAndLum(base[base_number], luminosity_byproduct)

    if rotate != 0:
        cur_map = np.rot90(cur_map, k=int(rotate))

    # cur_map = np.zeros(cur_map.shape)
    # cur_lum = np.zeros(cur_lum.shape)

    if pre_pool > 1:
        if len(cur_map)%pre_pool == 0:
            cur_map = block_reduce(cur_map, (pre_pool, pre_pool, pre_pool_z), np.sum)
        else:
            # I feel like I put this here for a reason...
            pass

    if log_input:
        # cur_map = np.log10(cur_map + 1e-6)
        # cur_map -= (-6)
        # cur_map = log_map(cur_map)
        cur_map = log_modulus(cur_map)

    # add gaussian noise
    if isinstance(make_map_noisy, (tuple, list, np.ndarray)):
        cur_map = add_noise_after_pool(cur_map, make_map_noisy, pre_pool, pre_pool_z)
    elif make_map_noisy > 0:
        cur_map = add_noise_after_pool(cur_map, make_map_noisy, pre_pool, pre_pool_z)

    # add in foregrounds
    if add_foregrounds:
        model_params = ModelParams()
        model_params.give_attributes(pre_pool=4, pre_pool_z=10)
        model_params.clean_parser_data()
        model_params.get_map_info(base[base_number] + '_map.npz')

        cur_map = add_foreground_noise(cur_map, model_params.pix_x, model_params.pix_y, model_params.omega_pix,
                                model_params.nu, pre_pool_z, random_foreground_params=random_foreground_params)

    # apply gaussian smoothing
    if gaussian_smoothing > 0:
        cur_map = apply_gaussian_smoothing(cur_map, gaussian_smoothing)

    if lum_func_size is not None:
        if lum_func_size >= 1:
            # lumData = lumData[::lum_func_size]
            cur_lum = cur_lum[:lum_func_size]
        else:
            cur_lum = cur_lum[lum_func_size:]

    # cur_map = np.zeros(cur_map.shape)
    # cur_lum = np.zeros(cur_lum.shape)

    # Handle 3D maps correctly
    if threeD:
        cur_map = cur_map.reshape(len(cur_map), len(cur_map[0]), len(cur_map[0][0]), 1)

    # expand the dimensions of the map and luminosity byproduct to work with the tensor of the model
    base_map = np.expand_dims(cur_map, axis=0)
    base_lum = np.expand_dims(cur_lum, axis=0)

    # make a prediction for the luminoisty byproduct for the given map
    cnn_lum = model.predict(tf.convert_to_tensor(base_map), steps=1)

    # convert negative values to just 0
    cnn_lum[cnn_lum < 0] = 0

    # return loss and other metric data about the CNN's output
    if evaluate:
        print('Error and MSE for the given base_number:')
        loss = model.evaluate(tf.convert_to_tensor(base_map), tf.convert_to_tensor(base_lum), steps=1, verbose=0)
        print(loss)
    else:
        loss = [0,0]

    # return the simulated luminosity byproduct and the one from the CNN
    return(cur_lum, cnn_lum, loss)

# test a single model against multiple maps
def test_model_multiple_times(model, base, luminosity_byproduct='log', threeD=False,
    evaluate=True, log_input=False, make_map_noisy=0, base_numbers=[], test_size=10):

    batch_size = 40

    # if multiple base numbers are given then get that many base names to test
    if len(base_numbers) > 0:
        base = base[base_numbers]

    # if the test_size is bigger then the number of bases being used then set the test size to that
    if len(base) < test_size:
        test_size = len(base)

    # setup dataset used for testing
    dataset = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(base))
    dataset = dataset.shuffle(buffer_size=len(base))
    dataset = dataset.map(lambda item: tuple(tf.py_func(utf8FileToMapAndLum, [item, luminosity_byproduct, threeD, log_input, make_map_noisy], [tf.float64, tf.float64])))
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)

    print(model.evaluate(dataset, steps=test_size, verbose=1))

# plot the results of a single CNN and map
def plot_model_test(cur_lum, cnn_lum, lumLogBinCents, y_label, lum_func_size=49):
    # # get moving average of CNN result to make it look smoother
    # window_size = 5
    # window = np.ones(window_size)/float(window_size)
    # avg = np.convolve(cnn_lum[0], window, 'same')

    # plot the simulated and CNN data
    plt.semilogx(lumLogBinCents[:lum_func_size], cnn_lum[0][:lum_func_size], label='CNN')
    # plt.semilogx(lumLogBinCents[2:], avg[2:], label='Smoothed CNN')
    plt.semilogx(lumLogBinCents[:lum_func_size], cur_lum[:lum_func_size], label='Simulated')
    plt.legend()
    plt.title('CNN Result')
    plt.xlabel('L (L_sun)')

    # handles axes correctly for the different byproducts
    if y_label == 'basic':
        plt.ylabel('dN/dL')
    elif y_label == 'basicL':
        plt.ylabel('dN/dL L')
    elif y_label == 'log':
        plt.ylabel('Log10(dN/dL)')
    elif y_label == 'numberCt':
        plt.ylabel('N')
    else:
        print('A bad y label was given, defaulting to basic')
        plt.ylabel('dN/dL')

    # show the plot
    plt.show()

    return()

# plot the ratio of a CNN's result to the underlying simulation
def plot_model_ratio(cur_lum, cnn_lum, lumLogBinCents, title, end_cut_off=1, lum_func_size=49):
    # get moving average of CNN result to make it look smoother
    # window_size = 5
    # window = np.ones(window_size)/float(window_size)
    # avg = np.convolve(cnn_lum[0], window, 'same')

    # plot the ratio
    ratio = cnn_lum[0][:lum_func_size]/cur_lum[:lum_func_size]
    lumLogBinCents = lumLogBinCents[:lum_func_size]
    # ratio_smooth = avg/cur_lum
    plt.semilogx(lumLogBinCents[:-end_cut_off], ratio[:-end_cut_off], label='CNN')
    # plt.semilogx(lumLogBinCents[2:-(end_cut_off)], ratio_smooth[2:-(end_cut_off)], label='Smoothed CNN')

    # handle the title correclty based on the byproduct used
    if title == 'basic':
        plt.title('Using dN/dL')
    elif title == 'basicL':
        plt.title('Using dN/dL L')
    elif title == 'log':
        plt.title('Using Log10(dN/dL)')
    elif y_label == 'numberCt':
        plt.title('Using N')
    else:
        print('A bad y label was given, defaulting to basic')
        plt.title('Using dN/dL')

    plt.ylabel('Predicted / Expected')
    plt.xlabel('L (L_sun)')
    plt.legend()

    plt.ylim([0.8,1.2])

    plt.show()

    return()

# plot multiple models together to compare them
def compare_multiple_models(model_keys, models_dict, base, base_number, lumLogBinCents,
    end_cut_off=1, evaluate=False, display='log', make_map_noisy=0):

    # lists to hold future luminosity values
    compare_lum = []
    cnn_lums = []

    print(display)

    # choose which converting function to use
    if display == 'log':
        converter = convert_lum_to_log
    elif display == 'basic':
        converter = convert_lum_to_basic
    elif display == 'numberCt':
        converter = convert_lum_to_numberCt
    else:
        converter = convert_lum_to_log

    # loop through each model and find the expected log luminosity function
    for key in model_keys:
        if evaluate:
            print(key)

        if 'log_in' in key:
            log_input = True
        else:
            log_input = False

        # test the model on the given map (base+base_number)
        cur_lum, cnn_lum = test_model(models_dict[key]['model'], base, base_number,
                                      luminosity_byproduct=models_dict[key]['luminosity_product'],
                                     threeD=models_dict[key]['threeD'], evaluate=evaluate,
                                     log_input=log_input, make_map_noisy=make_map_noisy)

        # append results to list
        cnn_lums.append(converter(cnn_lum[0], models_dict[key]['luminosity_product'], lumLogBinCents))

        # get the underlying log luminosity function once
        if len(compare_lum) == 0:
            compare_lum = converter(cur_lum, models_dict[key]['luminosity_product'], lumLogBinCents)

        # if requested to manual calculation of different loss metrics
        if evaluate:
            # print(cnn_lums[-1])
            print(logcosh(compare_lum, cnn_lums[-1]))
            print(logcosh_rel(compare_lum, cnn_lums[-1]))

    # plot the raw values and ratio of values together
    plot_multiple_models(compare_lum, cnn_lums, model_keys, lumLogBinCents, display=display)
    plot_multiple_models_ratios(compare_lum, cnn_lums, model_keys, lumLogBinCents, end_cut_off)

    return()

# plot multiple models together to compare them
def plot_multiple_models(compare_lum, cnn_lums, model_keys, lumLogBinCents, display='log'):
    plt.figure(figsize=(12, 6))

    # plot the underlying distribution
    plt.semilogx(lumLogBinCents, compare_lum, label='Simulated')

    # plot each individual model
    for i in range(len(cnn_lums)):
        plt.semilogx(lumLogBinCents, cnn_lums[i], label=model_keys[i])

    # basic plot stuff
    plt.legend()
    plt.title('CNN Result')
    plt.xlabel('L (L_sun)')

    if display == 'log':
        plt.ylabel('Log10(dN/dL)')
        plt.ylim(0, 5)
    if display == 'basic':
        plt.ylabel('dN/dL')
        # plt.ylim(0, 5)
    elif display == 'numberCt':
        plt.ylabel('Log10(N)')
        plt.ylim(0, 6)
    else:
        plt.ylabel('Log10(dN/dL)')
        plt.ylim(0, 5)
    plt.show()

    return()

# plot mutliple model ratios together for comparisons
def plot_multiple_models_ratios(compare_lum, cnn_lums, model_keys, lumLogBinCents, end_cut_off=1):
    plt.figure(figsize=(12, 6))

    # plot the ratio of 1
    ratio = compare_lum/compare_lum
    plt.semilogx(lumLogBinCents[:-end_cut_off], ratio[:-end_cut_off], label='100%')

    # plot the ratio for each model
    for i in range(len(cnn_lums)):
        ratio = cnn_lums[i]/compare_lum
        plt.semilogx(lumLogBinCents[:-end_cut_off], ratio[:-end_cut_off], label=model_keys[i])

    # basic plot stuff
    plt.legend()
    plt.title('Result')
    plt.ylabel('Predicted / Expected')
    plt.xlabel('L (L_sun)')
    # plt.ylim(0.85, 1.05)
    plt.show()

    return()

# convert the given luminosity byproduct to log luminosity function
def convert_lum_to_log(lum, luminosity_product, lumLogBinCents):
    # don't change log valued ones
    if luminosity_product == 'log':
        pass
    # take the log of the basic luminosity function
    elif luminosity_product == 'basic':
        lum = np.log10(lum)
    # divide by L and take the log for \phi L
    elif luminosity_product == 'basicL':
        lum = np.log10(lum/lumLogBinCents)
    # convert from N to \phi
    elif luminosity_product == 'numberCt':
        new_lum = [0]*len(lum)
        new_lum[-1] = 10**lum[-1]
        for i in reversed(range(len(lum)-1)):
            new_lum[i] = 10**lum[i] - 10**lum[i+1]
        lum = np.log10(new_lum)
    else:
        print('You shouldn\'t be here...')

    return(lum)

# convert the given luminosity byproduct to basic luminosity function
def convert_lum_to_basic(lum, luminosity_product, lumLogBinCents):
    # take log value to the power of 10
    if luminosity_product == 'log':
        lum = 10**lum
    # don't change the basic luminosity function
    elif luminosity_product == 'basic':
        pass
    # divide by L and take the log for \phi L
    elif luminosity_product == 'basicL':
        lum = lum/lumLogBinCents
    # convert from N to \phi
    elif luminosity_product == 'numberCt':
        new_lum = [0]*len(lum)
        new_lum[-1] = lum[-1]
        for i in reversed(range(len(lum)-1)):
            new_lum[i] = lum[i] - lum[i+1]
        lum = np.log10(new_lum)
    else:
        print('You shouldn\'t be here...')

    return(lum)

# convert the given luminosity byproduct to log number count
def convert_lum_to_numberCt(lum, luminosity_product, lumLogBinCents):
    # undo log, add together and then relog
    if luminosity_product == 'log':
        new_lum = [0]*len(lum)
        new_lum[-1] = 10**(lum[-1])
        for i in reversed(range(len(lum)-1)):
            new_lum[i] = 10**lum[i] + new_lum[i+1]
        lum = np.log10(new_lum)
    # take the log of the basic luminosity function and add together
    elif luminosity_product == 'basic':
        new_lum = [0]*len(lum)
        new_lum[-1] = lum[-1]
        for i in reversed(range(len(lum)-1)):
            new_lum[i] = lum[i] + new_lum[i+1]
        lum = np.log10(new_lum)
    # divide by L and then sum
    elif luminosity_product == 'basicL':
        lum = np.log10(lum/lumLogBinCents)
        new_lum = [0]*len(lum)
        new_lum[-1] = lum[-1]/lumLogBinCents[-1]
        for i in reversed(range(len(lum)-1)):
            new_lum[i] = lum[i]/lumLogBinCents[i] + new_lum[i+1]
        lum = np.log10(new_lum)
    # do nothing
    elif luminosity_product == 'numberCt':
        pass
    else:
        print('You shouldn\'t be here...')

    return(lum)

# convert a luminosity byproduct into log luminosity byproduct
def convert_lum_to_log(lum, luminosity_product, lumLogBinCents):
    # don't worry if it is already log
    if luminosity_product == 'log':
        pass
    # take the log if it is \phi
    elif luminosity_product == 'basic':
        lum = np.log10(lum)
    # divide by the luminosity and take the log if it is \phi*L
    elif luminosity_product == 'basicL':
        lum = np.log10(lum/lumLogBinCents)
    else:
        print('You shouldn\'t be here...')

    return(lum)

# log cosh loss function
def logcosh(y_true, y_pred):
    diff = np.log(np.cosh(y_true - y_pred))
    non_inf_diff = []
    for x in diff:
        # be careful of infinnities and nans
        if x != float('inf') and not np.isnan(x):
            non_inf_diff.append(x)

    # return average logcosh of difference between true and predicted value that isn't inf or nan
    return(sum(non_inf_diff)/len(non_inf_diff))

# relative log cosh loss function
def logcosh_rel(y_true, y_pred):
    diff = np.log(np.cosh((y_true - y_pred)/y_true))
    non_inf_diff = []
    for x in diff:
        # be careful of infinnities and nans
        if x != float('inf') and not np.isnan(x):
            non_inf_diff.append(x)
    diff = non_inf_diff

     # return average logcosh or relative error value that isn't inf or nan
    return(sum(non_inf_diff)/len(non_inf_diff))

# gets the expected output and ratios of expected output for a single model on multiple maps
def get_model_ratios(model, base, base_numbers, lum_func_size=49, luminosity_byproduct='log',
    threeD=False, evaluate=False, log_input=False, make_map_noisy=0, pre_pool=1, pre_pool_z=25,
    add_foregrounds=False, random_foreground_params=False):

    # lists to store results
    simulated_lums = np.zeros([len(base_numbers), lum_func_size])
    cnn_lums = np.zeros([len(base_numbers), lum_func_size])
    ratio_of_lums = np.zeros([len(base_numbers), lum_func_size])
    real_space_ratio_of_lums = np.zeros([len(base_numbers), lum_func_size])

    # get the output for each map
    for i, b in enumerate(base_numbers):
        temp_sim_lum, temp_cnn_lum, temp_loss = test_model(model, base, b, luminosity_byproduct=luminosity_byproduct, threeD=threeD,
                    evaluate=evaluate, log_input=log_input, make_map_noisy=make_map_noisy,
                    pre_pool=pre_pool, pre_pool_z=pre_pool_z, lum_func_size=lum_func_size,
                    add_foregrounds=add_foregrounds, random_foreground_params=random_foreground_params)

        # store output and make ratio list
        simulated_lums[i] = temp_sim_lum
        cnn_lums[i] = temp_cnn_lum
        ratio_of_lums[i] = (temp_cnn_lum/temp_sim_lum)[0]
        # real_space_ratio_of_lums[i]

    return(simulated_lums, cnn_lums, ratio_of_lums)

# function to give the stds of error for given ratio arrays
def std_of_model(ratio_of_lums):
    stds_upper = np.zeros(len(ratio_of_lums[0]))
    stds_lower = np.zeros(len(ratio_of_lums[0]))

    # find the std of error but use 1 as the mean instead of the actual mean
    for i in range(len(ratio_of_lums[0])):
        std_upper = 0
        std_lower = 0
        for j in ratio_of_lums[:,i]:
            if j > 1:
                std_upper += (j - 1)**2
            else:
                std_lower += (j - 1)**2
        stds_upper[i] = np.sqrt(std_upper/(len(ratio_of_lums[:,i])-1))
        stds_lower[i] = np.sqrt(std_lower/(len(ratio_of_lums[:,i])-1))

    return(stds_lower, stds_upper)

# # function to plot rough 95% contour around perfect prediction for different models
# def prediction_contour_plot(model_labels, model_lowers, model_uppers, model_lums, ratio_type='log', y_range=[0.7, 1.3], white_noise=0):
#     plt.figure(figsize=(12, 6))

#     plt.semilogx(model_lums[0], model_lowers[0]/model_lowers[0], label='100%')

#     for i in range(len(model_labels)):
#         plt.fill_between(model_lums[i], 1-2*model_lowers[i], 1+2*model_uppers[i], label=model_labels[i], alpha=0.15)

#     if len(y_range) == 2:
#         plt.ylim(y_range)
#     plt.xscale('log')
#     plt.xlabel('L (L_sun)')
#     if ratio_type == 'log':
#         plt.ylabel('Ratio of log10(dN/dL L)')
#     elif ratio_type == 'full':
#         plt.ylabel('Ratio of dN/dL L')
#     else:
#         plt.ylabel('Ratio of log10(dN/dL L)')
#     title = 'Power Spectrum Determination of dN/dL L Ratios'
#     if white_noise:
#         title += ' with {0} \\mu K noise'.format(white_noise)
#     plt.title(title)
#     plt.legend()

#     plt.show()

##########################################
# Load Power Spectrum Results
##########################################
# load power spectrum results from a given file name
def load_power_data(file_name):
    data = np.loadtxt(file_name, delimiter=',')
    lum = sorted(data[:,0])
    phi = list(sorted(data[:,1], reverse=True))

    return(lum, phi)

# interpolate a 1d function from a min value to a max value in log space
def interpolate_power_data(lum, phi, min_val, max_val):

    # do the interpolation
    f = interp1d(lum, phi)

    # get value of function at specific values in log space
    new_lum = np.logspace(min_val,max_val,50)
    new_phi = f(new_lum)

    return(new_lum, new_phi)

##########################################
# Intrinsic Scatter of luminosity functions
##########################################
# load luminositie function from a map
def load_lums(base, lumLogBinCents):
    lums = np.zeros([len(base), len(lumLogBinCents)])
    for i, b in enumerate(base):
        lums[i] = loadData(b + '_lum.npz')['lumFunc']

    return(lums)

# get the variance at each luminosity of the luminosity function for all maps
def get_means_vars(lums):
    means = np.zeros(len(lums[0]))
    std = np.zeros(len(lums[0]))

    for i, val in enumerate(lums[0]):
        means[i] = np.mean(lums[:,i])
        std[i] = np.std(lums[:,i])

    return(means, std)

##########################################
# Variance of CNN Tests
##########################################
# class to hold information about the results of the CNN testing a specific scenario
# contiains info about what it tested and statistics of the relative error of the luminosity function
class Prediction:
    # init and set values for basic attributes
    def __init__(self, predictions, label, conversion=1):
        self.predictions = predictions
        self.label = label
        self.keys = np.array(list(predictions.keys()))
        self.lum_func_size = len(self.predictions[self.keys[0]][0])

        self.res_ratio = []
        self.res_mean = []
        self.res_std = []

        self.res_median = []
        self.res_conf_interval = []

        self.conversion = conversion
        self.transformed_res_ratio = []
        self.transformed_res_mean = []
        self.transformed_res_std = []

        self.transformed_res_median = []
        self.transformed_res_conf_interval = []

        self.res_median_small = []
        self.res_conf_interval_small = []
        self.transformed_res_median_small = []
        self.transformed_res_conf_interval_small = []

        # calculate residual
        self.calculate_res()
        # calculate the mean and std of residual
        self.calculate_mean_std()
        # determine the 95% confidence interval of the test
        self.res_conf_interval, self.res_median, self.transformed_res_conf_interval, self.transformed_res_median = self.calculate_confidence_interval()

        # determine the 68% confidence interval of the test
        self.res_conf_interval_small, self.res_median_small, self.transformed_res_conf_interval_small, self.transformed_res_median_small = self.calculate_confidence_interval(conf=0.68)

    # get the residuals (relative error) of the luminosity function of a test
    def calculate_res(self):
        # avoid dividing by 0
        res_ratio = np.zeros([len(self.keys), len(self.predictions[self.keys[0]][0])])
        # loop over each map tested and each luminosity in said map
        for i, key in enumerate(self.keys):
            for j, val in enumerate(self.predictions[key][0]):
                if self.predictions[key][1][j] == 0.0 and self.predictions[key][0][j] != 0.0:
                    res_ratio[i,j] = None
                else:
                    res_ratio[i,j] = (self.predictions[key][0][j]-self.predictions[key][1][j]) / self.predictions[key][1][j]

        # get results out of log space
        transformed_res_ratio = np.zeros([len(self.keys), len(self.predictions[self.keys[0]][0])])
        for i, key in enumerate(self.keys):
            for j, val in enumerate(self.predictions[key][0]):
                transformed_res_ratio[i,j] = (10**self.predictions[key][0][j]-10**self.predictions[key][1][j]) / 10**self.predictions[key][1][j]

        # store values in attributes
        self.res_ratio = res_ratio
        self.transformed_res_ratio = transformed_res_ratio

        return()

    # calculate the mean and std of the residual (relative error)
    def calculate_mean_std(self):
        # be careful of nans
        self.res_mean = np.nanmean(self.res_ratio, 0)
        self.res_std = np.nanstd(self.res_ratio, 0)

        self.transformed_res_mean = np.nanmean(self.transformed_res_ratio, 0)
        self.transformed_res_std = np.nanstd(self.transformed_res_ratio, 0)

        return()

    # confidence interval centered on median
    def calculate_confidence_interval(self, conf=0.95):
        res_conf_interval = []
        res_median = []
        transformed_res_conf_interval = []
        transformed_res_median = []

        # do calculation of confidence interval for normal (unlogged) data
        for i, key in enumerate(self.res_ratio[0]):
            # sort list of residuals for a given luminosity
            sorted_res = np.array(sorted(self.res_ratio[:,i]))
            # remove nans if needed
            sorted_res = sorted(sorted_res[~np.isnan(sorted_res)])

            # get length of list to remove 5%
            map_numb = len(sorted_res)
            remove_maps = int(map_numb * (1-conf)/2)

            # store values in attributes
            # keep track of lower limit and upper limit of confidence interval
            res_conf_interval.append([sorted_res[remove_maps], sorted_res[-remove_maps]])
            res_median.append(sorted_res[int(map_numb/2)])

        res_conf_interval = np.array(res_conf_interval)
        res_median = np.array(res_median)

        # do calculation of confidence interval for logged data
        for i, key in enumerate(self.transformed_res_ratio[0]):
            # sort list of residuals for a given luminosity
            sorted_res = np.array(sorted(self.transformed_res_ratio[:,i]))
            # remove nans if needed
            sorted_res = sorted(sorted_res[~np.isnan(sorted_res)])

            # get length of list to remove 5%
            map_numb = len(sorted_res)
            remove_maps = int(map_numb * (1-conf)/2)

            transformed_res_conf_interval.append([sorted_res[remove_maps], sorted_res[-remove_maps]])
            transformed_res_median.append(sorted_res[int(map_numb/2)])

        transformed_res_conf_interval = np.array(transformed_res_conf_interval)
        transformed_res_median = np.array(transformed_res_median)

        return(res_conf_interval, res_median,
        transformed_res_conf_interval, transformed_res_median)

# plot an indivual CNN test result 95% relative error confidence error on an axis
def plot_res_contour(ax, res_pred, lumLogBinCents, alpha=0.25, color=None, label=None, conf=2):

    # put the confidence interval values in an error
    if conf == 1:
        conf_interval = np.array([res_pred.transformed_res_conf_interval_small[:,0], res_pred.transformed_res_conf_interval_small[:,1]])
    else:
        conf_interval = np.array([res_pred.transformed_res_conf_interval[:,0], res_pred.transformed_res_conf_interval[:,1]])

    # make sure there is a label to use
    if label is None:
        label = res_pred.label

    # use the specific color if given and make shaded region for contour
    if color is not None:
        p = ax.fill_between(lumLogBinCents[:res_pred.lum_func_size], conf_interval[0],
                         conf_interval[1], alpha=alpha, label=label, facecolor=color)
    else:
        p = ax.fill_between(lumLogBinCents[:res_pred.lum_func_size], conf_interval[0],
                         conf_interval[1], alpha=alpha, label=label)
    # get color and make clear boundaries for the contour
    color = p.get_facecolor()
    ax.plot(lumLogBinCents[:res_pred.lum_func_size], conf_interval[1], color=color[0][:-1], linewidth=2.5)
    ax.plot(lumLogBinCents[:res_pred.lum_func_size], conf_interval[0], color=color[0][:-1], linewidth=2.5)

    return(ax)

# plot a confidence interval that isn't a CNN test result
def plot_outside_contour(ax, conf_interval, lumLogBinCents, label, alpha=0.25, color=None):
    # use the specific color if given and make shaded region for contour
    if color is not None:
        p = ax.fill_between(lumLogBinCents, conf_interval[0],
                         conf_interval[1], alpha=alpha, label=label, facecolor=color)
    else:
        p = ax.fill_between(lumLogBinCents, conf_interval[0],
                         conf_interval[1], alpha=alpha, label=label)
    # get color and make clear boundaries for the contour
    color = p.get_facecolor()
    ax.plot(lumLogBinCents, conf_interval[1], color=color[0][:-1], linewidth=2.5)
    ax.plot(lumLogBinCents, conf_interval[0], color=color[0][:-1], linewidth=2.5)

    return(ax)

# plot the 95% confidence interval of the relative error for given CNN tests
def plot_res_contour_full(res_list, lumLogBinCents, alpha=0.25, colors=None, lum_points=False,
    plot_range=None, figsize=(18,9), ax=None, lum_point_size=100, conf=2):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # use a specific color if given
    for i, res_pred in enumerate(res_list):
        if colors is not None:
            color = colors[i]
        else:
            color = None

        # plot specific contour on axis for a given test
        plot_res_contour(ax, res_pred, lumLogBinCents, alpha=alpha, color=color, conf=conf)

    # asked also plot the luminosity points considered with red '+'
    if lum_points:
        ax.scatter(lumLogBinCents, [0]*len(lumLogBinCents), s=lum_point_size, marker='+', zorder=10, color='r',
                   label='Used Luminosity Points')

    # set the y-lim with the default or as given
    if plot_range is None:
    #     ax.set_ylim([-0.2, 0.2])
        ax.set_ylim([-1, 3])
    else:
        ax.set_ylim(plot_range)

    ax.set_xscale('log')
    ax.set_xlabel('L (L_sun)')

    return(ax)
