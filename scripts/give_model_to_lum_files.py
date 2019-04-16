import numpy as np

from limlam_mocker import limlam_mocker as llm
import lnn as lnn

# get model from file name
def model_from_file_name(file_name):
    if 'Li' in file_name:
        model = 'Li'
    elif 'Padmanabhan' in file_name:
        model = 'Padmanabhan'
    elif 'Breysse' in file_name:
        model = 'Breysse'
    else:
        model = 'None'
        
    return(model)

# for each map made go through and make a luminosity function file with info
mapLoc = '../maps2/'
maps = ['basic_Li', 'basic_Padmanabhan', 'different_maps', 'noisy_Li', 'random_Li', 'random_maps1', 'random_noisy_Li']

for mp in maps:
    # check for made maps
    map_loc = mapLoc + mp + '/'
    subFields = lnn.loadBaseFNames(map_loc)

    for sub in subFields:
        file_name = map_loc + sub + '_lum.npz'
        model = model_from_file_name(file_name)

        # get the luminosity function info
        data = lnn.loadData(file_name)

        # save the luminosity function info
        np.savez(str(file_name),
            logBinCent  =   data['logBinCent'],
            numberCt    =   data['numberCt'],
            lumFunc     =   data['lumFunc'],
            model       =   model)
