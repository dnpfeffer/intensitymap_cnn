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
    data = lumData['lumFunc']

    return(data)

### function to take the lumInfo and get just the luminosity function from it
def justLumFunc(lumData):
	lumData2 = []
	for l in lumData:
		lumData2.append(float(l))

	return(np.array(lumData2))

### function to take the luminosity function and take the log10 of it
### convert 0's to 1's so that there are no nans
def logLumFunc(lumData):
    logLumData = []
    for l in lumData:
        if l == 0:
            logLumData.append(0)
        else:
            logLumData.append(np.log10(l))

    return(np.array(logLumData))

### function to convert a basename into the map and map_cube and the makeLumFunc
def fileToMapAndLum(fName):
    maps, lumInfo = loadMapAndLum(fName)
    mapData = maps['map_cube'].astype(np.float32)
    lumData = justLumFunc(lumInfo['lumFunc']).astype(np.float32)

    return(mapData, lumData)

### function to convert a basename into the map and map_cube and the makeLogLumFunc
def fileToMapAndLogLum(fName):
    maps, lumInfo = loadMapAndLum(fName)
    mapData = maps['map_cube'].astype(np.float32)
    lumData = logLumFunc(lumInfo['lumFunc']).astype(np.float32)

    return(mapData, lumData)

### function to convert a utf-8 basename into the map map_cube and the makeLumFunc
def utf8FileToMapAndLum(fName):
    mapData, lumData = fileToMapAndLum(fName.decode('utf-8'))

    return(mapData, lumData)

### function to convert a utf-8 basename into the map map_cube and the makeLumFunc
def utf8FileToMapAndLum3D(fName):
    mapData, lumData = fileToMapAndLum(fName.decode('utf-8'))
    ### make sure to reshape the map data for the 3D convolutions
    mapData = mapData.reshape(len(mapData), len(mapData[0]), len(mapData[0][0]), 1)

    return(mapData, lumData)

### function to convert a utf-8 basename into the map map_cube and the makeLogLumFunc
def utf8FileToMapAndLogLum(fName):
    mapData, lumData = fileToMapAndLogLum(fName.decode('utf-8'))

    return(mapData, lumData)

### function to convert a utf-8 basename into the map map_cube and the makeLogLumFunc
def utf8FileToMapAndLogLum3D(fName):
    mapData, lumData = fileToMapAndLogLum(fName.decode('utf-8'))
    ### make sure to reshape the map data for the 3D convolutions
    mapData = mapData.reshape(len(mapData), len(mapData[0]), len(mapData[0][0]), 1)

    return(mapData, lumData)
