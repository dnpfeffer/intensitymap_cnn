import numpy as np
#import tensorflow as tf

from .ioFuncs import *

### needed to save the map
from limlam_mocker import limlam_mocker as llm

### function to convert a map file into a mapcube for the cnn
def fileToMapData(fName):
    mapData = loadData(fName)
    data = mapData['map_cube']

    return(data)

### function to return the luminosity function from a lum files
def fileToLum(fName):
    lumData = loadData(fName)
    data = lumData

    return(data)

### function to return the required luminosity function byproduct
def lumFuncByproduct(lumInfo, lumByproduct='basic'):
    lumResult = []
    for i in range(len(lumInfo['lumFunc'])):
        l = lumInfo['lumFunc'][i]
        if lumByproduct == 'basic':
            lumResult.append(float(l))
        elif lumByproduct == 'log':
            if l == 0:
                lumResult.append(0.0)
            else:
                lumResult.append(np.log10(float(l)))
        elif lumByproduct == 'basicL':
            lumResult.append(float(l * lumInfo['logBinCent'][i]))
        else:
            pass

    return(np.array(lumResult))

### function to convert a basename into the map map_cube and the wanted luminosity byproduct
def fileToMapAndLum(fName, lumByproduct='basic'):
    maps, lumInfo = loadMapAndLum(fName)
    mapData = maps['map_cube']
    lumData = lumFuncByproduct(lumInfo, lumByproduct)

    return(mapData, lumData)

### function to convert a utf-8 basename into the map map_cube and the luminosity byproduct
def utf8FileToMapAndLum(fName, lumByproduct='basic', ThreeD=False):
    lumByproduct = lumByproduct.decode("utf-8")
    mapData, lumData = fileToMapAndLum(fName.decode('utf-8'), lumByproduct)

    if ThreeD:
        ### make sure to reshape the map data for the 3D convolutions
        mapData = mapData.reshape(len(mapData), len(mapData[0]), len(mapData[0][0]), 1)

    return(mapData, lumData)
