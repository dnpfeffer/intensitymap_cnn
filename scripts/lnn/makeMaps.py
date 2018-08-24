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
    vals, bins = np.histogram(lco, bins=np.logspace(0,7, 50))

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

### make a paramter object from a dict
class params(object):
    def __init__(self,d):
        self.__dict__ = d

### make a default parameter object
def defaultParams():
    paramsDict = {}
    catName = 'COMAP_z2.39-3.44_1140Mpc_seed_13579_rotate_0_subfield_0.npz'
    ### get the map and catalog output file name
    paramsDict['halo_catalogue_file'] = '../catalogues/'+ catName
    paramsDict['map_output_file'] = '../maps/' + catName

    ### halo luminosity function
    paramsDict['model'] = 'Li'
    paramsDict['coeffs'] = [0.0, 1.37, -1.74, 0.3, 0.3]
    paramsDict['min_mass'] = 2.5e10

    ### map parameters
    paramsDict['nu_rest'] = 115.27
    paramsDict['nu_i'] = 34.
    paramsDict['nu_f'] = 26.
    paramsDict['nmaps'] = 100
    paramsDict['fov_x'] = 1.4
    paramsDict['fov_y'] = 1.4
    paramsDict['npix_x'] = 256
    paramsDict['npix_y'] = 256

    ### Plot parameters
    paramsDict['plot_cube'] = True
    paramsDict['plot_pspec'] = True

    param = params(paramsDict)

    return(param)

### function to get non-default parameters
### model and coeffs have default values, but the other ones do not
def getParams(haloCat, mapFile, model='Li', coeffs=None, **kwargs):
    params = defaultParams()
    params.halo_catalogue_file = haloCat
    params.map_output_file = mapFile
    params.model=model
    params.coeffs=coeffs

    if kwargs is not None:
        for key, value in kwargs.items():
            if key in params.__dict__:
                params.__dict__[key] = value
            else:
                print('Field {} does not exist in the parameter object'.format(key))

    return(params)
