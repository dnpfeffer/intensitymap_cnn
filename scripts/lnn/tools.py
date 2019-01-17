import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .ioFuncs import *
from .preprocessing import *

import tensorflow as tf
from tensorflow import keras as k

### gets the number of times a model has been trained
def get_model_iteration(model_name, model_matches=[], model_loc=[]):
    ### handle the possible inputs
    ### can be given a list of names or the directory with the model files
    if len(model_matches) > 0 and len(model_loc) > 0:
        print('Given both a modelLoc and model match.  Defaulting to model match')
    elif len(model_loc) > 0 and len(model_matches) == 0:
        model_matches = get_model_name_matches(model_loc, model_name)

    ### yell at the user if there was no real models with that name
    if len(model_matches) == 0 and len(model_loc) == 0:
        print('Either no model locations or model matches were given or there was not a completed run with the given model name')
        return(0)

    ### get the number of models with that name with completed histories
    ct = 0
    for m in model_matches:
        if model_name + '_history' in m:
            ct += 1

    ### return the count
    return(ct)

### gets the total history of a model
def get_full_history(model_name, model_loc):
    ### get the nubmer of times the model has been trained
    train_count = get_model_iteration(model_name, model_loc=model_loc)

    ### load up the first trianing history
    history_name = model_loc + model_name + '_history'
    base_history = load_history(history_name)

    ### return the history if it was only trained once
    if train_count == 1:
        return(base_history)

    ### combine histories if the model was trained multiple times
    for i in range(1, train_count):
        new_history = load_history('{0}_{1}'.format(history_name, i))
        for key in base_history:
            base_history[key] = base_history[key] + new_history[key]

    return(base_history)

### convert a luminosity byproduct into log luminosity byproduct
def convert_lum_to_log(lum, luminosity_product, lumLogBinCents):
    ### don't worry if it is already log
    if luminosity_product == 'log':
        pass
    ### take the log if it is \phi
    elif luminosity_product == 'basic':
        lum = np.log10(lum)
    ### divide by the luminosity and take the log if it is \phi*L
    elif luminosity_product == 'basicL':
        lum = np.log10(lum/lumLogBinCents)
    else:
        print('You shouldn\'t be here...')

    return(lum)

### handle booleans for argument parsing
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

### make a file name from the given model information
def make_file_name(luminosity_byproduct, numb_layers, ThreeD, base_filters):
    if luminosity_byproduct == 'log':
        lb_string = 'log'
    elif luminosity_byproduct == 'basic':
        lb_string = 'full'
    elif luminosity_byproduct == 'basicL':
        lb_string = 'fullL'
    else:
        print('There should not be a way for someone to be in make_file_name without a valid luminosity_byproduct: {0}'.format(luminosity_byproduct))
        exit(0)

    if ThreeD:
        ThreeD_string = '3D'
    else:
        ThreeD_string = '2D'

    file_name = '{0}_lum_{1}_layer_{2}_{3}_filters_model'.format(lb_string, numb_layers, ThreeD_string, base_filters)

    return(file_name)


###############################################################
### Jupyter Notebook Plotting Tools ###########################
###############################################################

### plots the history of training a model and compares two metrics at the same time
def history_compare_two_metrics(history, metrics=['loss', 'mean_squared_error'], start_loc=1):
    ### set up plot
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax_list = [ax1, ax2]
    colors = ['r', 'b']

    ### don't allow more then 2 metrics to be given
    if len(metrics) > 2:
        print("More then 2 metrics were given.  Please only give 2")
        return()

    ### do the plot for each metric given
    for i in range(len(metrics)):
        key = metrics[i]
        ax = ax_list[i]
        color = colors[i]

        ### actual plotting on each axis
        ax.semilogy(range(len(history[key]))[start_loc:], history[key][start_loc:], label=key, color=color)
        ax.semilogy(range(len(history['val_' + key]))[start_loc:], history['val_' + key][start_loc:], color=color, ls='--')
        ax.set_ylabel(key, color=color)
        ax.tick_params('y', which='both', labelcolor=color)

    ### set up the legend to show training and validation line types
    custom_lines = [Line2D([0], [0], linestyle='-', color='k', lw=1),
                Line2D([0], [0], linestyle='--', color='k', lw=1)]
    ax1.legend(custom_lines, ['Training', 'Validation'])

    ### display the plot
    ax1.set_xlabel('Epoch')
    fig.tight_layout()
    plt.show()

    return()

### get the predicted and simulated luminosity function byproduct for a given model and map
def test_model(model, base, base_number, luminosity_byproduct='log', threeD=False,
                evaluate=True, log_input=False, make_map_noisy=0):
    ### get the simulated map and luminosity byproduct
    cur_map = fileToMapData(base[base_number] + '_map.npz', log_input=log_input)
    cur_lum = lumFuncByproduct(fileToLum(base[base_number] + '_lum.npz'), luminosity_byproduct)

    ### add gaussian noise, but make sure it is positive valued
    if make_map_noisy > 0:
        cur_map = cur_map + np.absolute(np.random.normal(0, make_map_noisy, cur_map.shape))

    ### Handle 3D maps correctly
    if threeD:
        cur_map = cur_map.reshape(len(cur_map), len(cur_map[0]), len(cur_map[0][0]), 1)

    ### expand teh dimensions of the map and luminosity byproduct to work with the tensor of the model
    base_map = np.expand_dims(cur_map, axis=0)
    base_lum = np.expand_dims(cur_lum, axis=0)

    ### make a prediction for the luminoisty byproduct for the given map
    cnn_lum = model.predict(tf.convert_to_tensor(base_map), steps=1)

    # print(cnn_lum)
    # print('testing')

    ### return loss and other metric data about the CNN's output
    if evaluate:
        print('Error and MSE for the given base_number:')
        print(model.evaluate(tf.convert_to_tensor(base_map), tf.convert_to_tensor(base_lum), steps=1, verbose=0))

    ### return the simulated luminosity byproduct and the one from the CNN
    return(cur_lum, cnn_lum)

### test a single model against multiple maps
def test_model_multiple_times(model, base, luminosity_byproduct='log',
                                threeD=False, evaluate=True, log_input=False,
                                make_map_noisy=0, base_numbers=[], test_size=10):

    batch_size = 40

    if len(base_numbers) > 0:
        base = base[base_numbers]

    if len(base) < test_size:
        test_size = len(base)

    dataset = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(base))
    dataset = dataset.shuffle(buffer_size=len(base))
    dataset = dataset.map(lambda item: tuple(tf.py_func(utf8FileToMapAndLum, [item, luminosity_byproduct, threeD, log_input, make_map_noisy], [tf.float64, tf.float64])))
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)

    print(model.evaluate(dataset, steps=test_size, verbose=1))

### plot the results of a single CNN and map
def plot_model_test(cur_lum, cnn_lum, lumLogBinCents, y_label):
    ### get moving average of CNN result to make it look smoother
    window_size = 5
    window = np.ones(window_size)/float(window_size)
    avg = np.convolve(cnn_lum[0], window, 'same')

    ### plot the simulated and CNN data
    plt.semilogx(lumLogBinCents, cnn_lum[0], label='CNN')
    plt.semilogx(lumLogBinCents[2:], avg[2:], label='Smoothed CNN')
    plt.semilogx(lumLogBinCents, cur_lum, label='Simulated')
    plt.legend()
    plt.title('CNN Result')
    plt.xlabel('L (L_sun)')

    ### handles axes correctly for the different byproducts
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

    ### show the plot
    plt.show()

    return()

### plot the ratio of a CNN's result to the underlying simulation
def plot_model_ratio(cur_lum, cnn_lum, lumLogBinCents, title, end_cut_off=1):
    ### get moving average of CNN result to make it look smoother
    window_size = 5
    window = np.ones(window_size)/float(window_size)
    avg = np.convolve(cnn_lum[0], window, 'same')

    ### plot the ratio
    ratio = cnn_lum[0]/cur_lum
    ratio_smooth = avg/cur_lum
    plt.semilogx(lumLogBinCents[:-end_cut_off], ratio[:-end_cut_off], label='CNN')
    plt.semilogx(lumLogBinCents[2:-(end_cut_off)], ratio_smooth[2:-(end_cut_off)], label='Smoothed CNN')

    ### handle the title correclty based on the byproduct used
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

    plt.show()

    return()

### plot multiple models together to compare them
def compare_multiple_models(model_keys, models_dict, base, base_number, lumLogBinCents,
end_cut_off=1, evaluate=False, display='log', make_map_noisy=0):

    ### lists to hold future luminosity values
    compare_lum = []
    cnn_lums = []

    print(display)

    ### choose which converting function to use
    if display == 'log':
        converter = convert_lum_to_log
    elif display == 'basic':
        converter = convert_lum_to_basic
    elif display == 'numberCt':
        converter = convert_lum_to_numberCt
    else:
        converter = convert_lum_to_log

    ### loop through each model and find the expected log luminosity function
    for key in model_keys:
        if evaluate:
            print(key)

        if 'log_in' in key:
            log_input = True
        else:
            log_input = False

        cur_lum, cnn_lum = test_model(models_dict[key]['model'], base, base_number,
                                      luminosity_byproduct=models_dict[key]['luminosity_product'],
                                     threeD=models_dict[key]['threeD'], evaluate=evaluate,
                                     log_input=log_input, make_map_noisy=make_map_noisy)

        cnn_lums.append(converter(cnn_lum[0], models_dict[key]['luminosity_product'], lumLogBinCents))

        ### get the underlying log luminosity function once
        if len(compare_lum) == 0:
            compare_lum = converter(cur_lum, models_dict[key]['luminosity_product'], lumLogBinCents)


        if evaluate:
            # print(cnn_lums[-1])
            print(logcosh(compare_lum, cnn_lums[-1]))
            print(logcosh_rel(compare_lum, cnn_lums[-1]))

    ### plot the raw values and ratio of values together
    plot_multiple_models(compare_lum, cnn_lums, model_keys, lumLogBinCents, display=display)
    plot_multiple_models_ratios(compare_lum, cnn_lums, model_keys, lumLogBinCents, end_cut_off)

### plot multiple models together to compare them
def plot_multiple_models(compare_lum, cnn_lums, model_keys, lumLogBinCents, display='log'):
    plt.figure(figsize=(12, 6))

    ### plot the underlying distribution
    plt.semilogx(lumLogBinCents, compare_lum, label='Simulated')

    ### plot each individual model
    for i in range(len(cnn_lums)):
        plt.semilogx(lumLogBinCents, cnn_lums[i], label=model_keys[i])

    ### basic plot stuff
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

### plot mutliple model ratios together for comparisons
def plot_multiple_models_ratios(compare_lum, cnn_lums, model_keys, lumLogBinCents, end_cut_off=1):
    plt.figure(figsize=(12, 6))

    ### plot the ratio of 1
    ratio = compare_lum/compare_lum
    plt.semilogx(lumLogBinCents[:-end_cut_off], ratio[:-end_cut_off], label='100%')

    ### plot the ratio for each model
    for i in range(len(cnn_lums)):
        ratio = cnn_lums[i]/compare_lum
        plt.semilogx(lumLogBinCents[:-end_cut_off], ratio[:-end_cut_off], label=model_keys[i])

    ### basic plot stuff
    plt.legend()
    plt.title('Result')
    plt.ylabel('Predicted / Expected')
    plt.xlabel('L (L_sun)')
    # plt.ylim(0.85, 1.05)
    plt.show()

    return()

### convert the given luminosity byproduct to log luminosity function
def convert_lum_to_log(lum, luminosity_product, lumLogBinCents):
    ### don't change log valued ones
    if luminosity_product == 'log':
        pass
    ### take the log of the basic luminosity function
    elif luminosity_product == 'basic':
        lum = np.log10(lum)
    ### divide by L and take the log for \phi L
    elif luminosity_product == 'basicL':
        lum = np.log10(lum/lumLogBinCents)
    ### convert from N to \phi
    elif luminosity_product == 'numberCt':
        new_lum = [0]*len(lum)
        new_lum[-1] = 10**lum[-1]
        for i in reversed(range(len(lum)-1)):
            new_lum[i] = 10**lum[i] - 10**lum[i+1]
        lum = np.log10(new_lum)
    else:
        print('You shouldn\'t be here...')

    return(lum)

### convert the given luminosity byproduct to basic luminosity function
def convert_lum_to_basic(lum, luminosity_product, lumLogBinCents):
    ### take log value to the power of 10
    if luminosity_product == 'log':
        lum = 10**lum
    ### don't change the basic luminosity function
    elif luminosity_product == 'basic':
        pass
    ### divide by L and take the log for \phi L
    elif luminosity_product == 'basicL':
        lum = lum/lumLogBinCents
    ### convert from N to \phi
    elif luminosity_product == 'numberCt':
        new_lum = [0]*len(lum)
        new_lum[-1] = lum[-1]
        for i in reversed(range(len(lum)-1)):
            new_lum[i] = lum[i] - lum[i+1]
        lum = np.log10(new_lum)
    else:
        print('You shouldn\'t be here...')

    return(lum)

### convert the given luminosity byproduct to log number count
def convert_lum_to_numberCt(lum, luminosity_product, lumLogBinCents):
    ### undo log, add together and then relog
    if luminosity_product == 'log':
        new_lum = [0]*len(lum)
        new_lum[-1] = 10**(lum[-1])
        for i in reversed(range(len(lum)-1)):
            new_lum[i] = 10**lum[i] + new_lum[i+1]
        lum = np.log10(new_lum)
    ### take the log of the basic luminosity function and add together
    elif luminosity_product == 'basic':
        new_lum = [0]*len(lum)
        new_lum[-1] = lum[-1]
        for i in reversed(range(len(lum)-1)):
            new_lum[i] = lum[i] + new_lum[i+1]
        lum = np.log10(new_lum)
    ### divide by L and then sum
    elif luminosity_product == 'basicL':
        lum = np.log10(lum/lumLogBinCents)
        new_lum = [0]*len(lum)
        new_lum[-1] = lum[-1]/lumLogBinCents[-1]
        for i in reversed(range(len(lum)-1)):
            new_lum[i] = lum[i]/lumLogBinCents[i] + new_lum[i+1]
        lum = np.log10(new_lum)
    ### do nothing
    elif luminosity_product == 'numberCt':
        pass
    else:
        print('You shouldn\'t be here...')

    return(lum)

### log cosh loss function
def logcosh(y_true, y_pred):
    diff = np.log(np.cosh(y_true - y_pred))
    non_inf_diff = []
    for x in diff:
        if x != float('inf') and not np.isnan(x):
            non_inf_diff.append(x)
    diff = non_inf_diff
    return(sum(non_inf_diff)/len(non_inf_diff))

### relative log cosh loss function
def logcosh_rel(y_true, y_pred):
    diff = np.log(np.cosh((y_true - y_pred)/y_true))
    non_inf_diff = []
    for x in diff:
        if x != float('inf') and not np.isnan(x):
            non_inf_diff.append(x)
    diff = non_inf_diff
    return(sum(non_inf_diff)/len(non_inf_diff))

### gets the expected output and ratios of expected output for a single model on multiple maps
def get_model_ratios(model, base, base_numbers, luminosity_length=49, luminosity_byproduct='log',
                threeD=False, evaluate=False, log_input=False, make_map_noisy=0):

    ### lists to store results
    simulated_lums = np.zeros([len(base_numbers), luminosity_length])
    cnn_lums = np.zeros([len(base_numbers), luminosity_length])
    ratio_of_lums = np.zeros([len(base_numbers), luminosity_length])
    real_space_ratio_of_lums = np.zeros([len(base_numbers), luminosity_length])

    ### get the output for each map
    for i, b in enumerate(base_numbers):
        temp_sim_lum, temp_cnn_lum = test_model(model, base, b, luminosity_byproduct=luminosity_byproduct, threeD=threeD,
                    evaluate=evaluate, log_input=log_input, make_map_noisy=make_map_noisy)

        ### store output and make ratio list
        simulated_lums[i] = temp_sim_lum
        cnn_lums[i] = temp_cnn_lum
        ratio_of_lums[i] = (temp_cnn_lum/temp_sim_lum)[0]
        real_space_ratio_of_lums[i]

    return(simulated_lums, cnn_lums, ratio_of_lums)

### function to give the stds of error for given ratio arrays
def std_of_model(ratio_of_lums):
    stds = np.zeros(len(ratio_of_lums[0]))

    ### find the std of error but use 1 as the mean instead of the actual mean
    for i in range(len(ratio_of_lums[0])):
        std = 0
        for j in ratio_of_lums[:,i]:
            std += (j - 1)**2
        stds[i] = np.sqrt(std/len(ratio_of_lums[:,i]))

    return(stds)

### function to plot rough 95% contour around perfect prediction for different models
def prediction_contour_plot(model_labels, model_lowers, model_uppers, model_lums, ratio_type='log', y_range=[0.7, 1.3], white_noise=0):
    plt.figure(figsize=(12, 6))

    plt.semilogx(model_lums[0], model_lowers[0]/model_lowers[0], label='100%')

    for i in range(len(model_labels)):
        plt.fill_between(model_lums[i], 1-2*model_lowers[i], 1+2*model_uppers[i], label=model_labels[i], alpha=0.15)

    if len(y_range) == 2:
        plt.ylim(y_range)
    plt.xscale('log')
    plt.xlabel('L (L_sun)')
    if ratio_type == 'log':
        plt.ylabel('Ratio of log10(dN/dL L)')
    elif ratio_type == 'full':
        plt.ylabel('Ratio of dN/dL L')
    else:
        plt.ylabel('Ratio of log10(dN/dL L)')
    title = 'Power Spectrum Determination of dN/dL L Ratios'
    if white_noise:
        title += ' with {0} \\mu K noise'.format(white_noise)
    plt.title(title)
    plt.legend()

    plt.show()
