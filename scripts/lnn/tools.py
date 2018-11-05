import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .ioFuncs import *

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
    for i in range(2, train_count+1):
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
