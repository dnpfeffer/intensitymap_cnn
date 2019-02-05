import numpy as np
#import tensorflow as tf

from skimage.measure import block_reduce

from .ioFuncs import *

### needed to save the map
from limlam_mocker import limlam_mocker as llm

### function to convert a map file into a mapcube for the cnn
def fileToMapData(fName, log_input=False):
    mapData = loadData(fName)
    data = mapData['map_cube']

    if log_input:
        data = np.log10(data + 1e-6)

    return(data)

### function to return the luminosity function from a lum files
def fileToLum(fName):
    lumData = loadData(fName)
    data = lumData

    return(data)

### function to return the required luminosity function byproduct
def lumFuncByproduct(lumInfo, lumByproduct='basic'):
    if lumByproduct != 'category':
        lumResult = []
        for i in range(len(lumInfo['lumFunc'])):
            if lumByproduct == 'basic':
                l = lumInfo['lumFunc'][i]
                lumResult.append(float(l))
            elif lumByproduct == 'log':
                l = lumInfo['lumFunc'][i]
                if l == 0:
                    lumResult.append(0.0)
                else:
                    lumResult.append(np.log10(float(l)))
            elif lumByproduct == 'basicL':
                l = lumInfo['lumFunc'][i]
                lumResult.append(float(l * lumInfo['logBinCent'][i]))
            elif lumByproduct == 'numberCt':
                n = lumInfo['numberCt'][i]
                lumResult.append(float(n))
            else:
                pass

        return(np.array(lumResult))
    else:
        return(lumModelToInt(lumInfo))

### functiont hat converts a luminosity model into an int for categorizing
def lumModelToInt(lumInfo):
    model = lumInfo['model']
    model_int = -1

    if model == 'Li':
        model_int = 0
    elif model == '':
        model_int = 1
    elif model == '':
        model_int = 2

    return(model_int)

### function to convert a basename into the map map_cube and the wanted luminosity byproduct
def fileToMapAndLum(fName, lumByproduct='basic'):
    maps, lumInfo = loadMapAndLum(fName)
    mapData = maps['map_cube']
    lumData = lumFuncByproduct(lumInfo, lumByproduct)

    return(mapData, lumData)

### function to convert a utf-8 basename into the map map_cube and the luminosity byproduct
def utf8FileToMapAndLum(fName, lumByproduct='basic', ThreeD=False, log_input=False,
                        make_map_noisy=0, pre_pool=1, pre_pool_z=1, lum_func_size=None):
    lumByproduct = lumByproduct.decode("utf-8")
    mapData, lumData = fileToMapAndLum(fName.decode('utf-8'), lumByproduct)

    #########################
    # very temp thing that needs to be done correctly later
    # make some maps have only zeros and the lum func be zero as well
    # randomly scale things that aren't zero
    ########################
    if np.random.rand() < 0.1:
        # mapData = np.zeros(mapData.shape)
        # lumData = np.zeros(lumData.shape)
        pass
    else:
        # lumLogBinCents = loadLogBinCenters(fName.decode('utf-8'))
        # # bin_index_diff = np.random.randint(0, len(lumLogBinCents)-1)
        # bin_index_diff = int(np.random.poisson(5))
        # mapData, lumData = scaleMapAndLum(
        #     mapData, lumData, lumLogBinCents, bin_index_diff)
        pass
    ########################
    #######################

    # add gaussian noise, but make sure it is positive valued
    if make_map_noisy > 0:
        mapData = mapData + \
            np.absolute(np.random.normal(0, make_map_noisy, mapData.shape))

    if pre_pool > 1:
        if len(mapData) % pre_pool == 0:
            mapData = block_reduce(
                mapData, (pre_pool, pre_pool, pre_pool_z), np.sum)
        else:
            pass

    if log_input:
        mapData = np.log10(mapData + 1e-6)

        mapData -= (np.min(mapData))

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
