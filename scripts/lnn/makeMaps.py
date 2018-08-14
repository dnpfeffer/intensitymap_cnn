import numpy as np

from .ioFuncs import *

### needed to make the map
from limlam_mocker import limlam_mocker as llm

### function to generate the luminosity function of a halo
def makeLumFunc(halos):
    ### remove any halos that have no luminosity
    index = np.argwhere(halos.Lco==0.0)
    lco = np.delete(halos.Lco, index)

    ### generate the histogram
    vals, bins = np.histogram(lco, bins=np.logspace(np.log10(min(lco)),np.log10(max(lco)), 50))

    ### needed arrays for the actual luminosity function
    lFunc = [0]*len(vals)
    logLCent = [0]*len(vals)
    # lCent = [0]*len(vals)

    ### go backwards through the histogram and higher luminosity values to lower ones
    ### also get the log center of the bins
    for i in reversed(range(len(vals))):
        if(len(vals)-1 == i):
            lFunc[i] = vals[i]
        else:
            lFunc[i] = lFunc[i+1] + vals[i]

        logLCent[i] = 10**((np.log10(bins[i]) + np.log10(bins[i+1]))/2)
        # lCent[i] = (bins[i] + bins[i+1])/2

    ### return bin centers and luminosity function values
    return([logLCent, lFunc])

### function to make a map given a set of parameters
def makeMapAndLumFunc(params, verbose=False):
    llm.debug.verbose = verbose

    if llm.debug.verbose:
        llm.write_time('Starting Line Intensity Mapper')

    ### Setup maps to output
    mapinst   = llm.params_to_mapinst(params);

    ### Load halos from catalogue
    halos, cosmo = llm.load_peakpatch_catalogue(params.halo_catalogue_file, verbose=llm.debug.verbose)
    halos        = llm.cull_peakpatch_catalogue(halos, params.min_mass, mapinst, verbose=llm.debug.verbose)

    ### Calculate Luminosity of each halo
    halos.Lco    = llm.Mhalo_to_Lco(halos, params.model, params.coeffs, verbose=llm.debug.verbose)

    ### generate the luminosity function
    lumInfo = makeLumFunc(halos)

    ### Bin halo luminosities into map
    mapinst.maps = llm.Lco_to_map(halos,mapinst, verbose=llm.debug.verbose)

    if llm.debug.verbose:
        llm.write_time('Finished Line Intensity Map Generation')

    return(mapinst, lumInfo)

### function to make and save a map given a set of parameters
def makeAndSaveMapAndLumFunc(params, verbose=False):
    llm.debug.verbose = verbose

    if llm.debug.verbose:
        llm.write_time('Starting Line Intensity Mapper')

    ### Setup maps to output
    mapinst   = llm.params_to_mapinst(params);

    ### Load halos from catalogue
    halos, cosmo = llm.load_peakpatch_catalogue(params.halo_catalogue_file, verbose=llm.debug.verbose)
    halos        = llm.cull_peakpatch_catalogue(halos, params.min_mass, mapinst, verbose=llm.debug.verbose)

    ### Calculate Luminosity of each halo
    halos.Lco    = llm.Mhalo_to_Lco(halos, params.model, params.coeffs, verbose=llm.debug.verbose)

    ### generate the luminosity function
    lumInfo = makeLumFunc(halos)

    ### Bin halo luminosities into map
    mapinst.maps = llm.Lco_to_map(halos,mapinst, verbose=llm.debug.verbose)

    if llm.debug.verbose:
        llm.write_time('Finished Line Intensity Map Generation')

    ### save the map and lum function
    saveMapAndLum(mapinst, lumInfo)
    return()

### function to make maps and luminosity functions for every subfield-catalog in a directory that isn't already made
def makeCatdMaps(params, catalogList, catLoc, mapLoc, verbose=False):
    paramsDict = {}

    baseNames = loadBaseFNames(mapLoc)

    for cat in catalogList:

        if cat[:-4] in baseNames:
            if verbose:
                print('The map and luminosity function for {} have already been made\n\tMoving on to the next catalog'.format(cat))
            continue

        paramsDict['halo_catalogue_file'] = catLoc + '/' + cat
        paramsDict['map_output_file'] = mapLoc + '/' + cat
        params = setParams(paramsDict, params)

        if verbose:
            print('\n\nAbout to generate map for catalog {}...'.format(cat))
        maps, lumInfo = makeMapAndLumFunc(params, verbose)
        saveMapAndLum(maps, lumInfo)

### modify the parameters object given a dict of the parametes
def setParams(paramsDict, params):
    for key, val in paramsDict.items():
        setattr(params, key, val)
    return(params)
