import numpy as np
import random

from .ioFuncs import *

# needed to make the map
from limlam_mocker import limlam_mocker as llm

# function to generate the luminosity function of a halo
def makeLumFunc(halos, model=None):
    # remove any halos that have no luminosity
    index = np.argwhere(halos.Lco==0.0)
    lco = np.delete(halos.Lco, index)

    # generate the histogram
    try:
        vals, bins = np.histogram(lco, bins=np.logspace(3.5,7, 50))
    except:
        print(lco, np.logspace(3.5,7, 50))
        exit(0)

    # needed arrays for the actual luminosity function
    lFunc = [0]*len(vals)
    logLCent = [0]*len(vals)
    # lCent = [0]*len(vals)

    # go backwards through the histogram and higher luminosity values to lower ones
    # also get the log center of the bins
    for i in reversed(range(len(vals))):
        if(len(vals)-1 == i):
            lFunc[i] = vals[i]
        else:
            lFunc[i] = lFunc[i+1] + vals[i]

        logLCent[i] = 10**((np.log10(bins[i]) + np.log10(bins[i+1]))/2)
        # lCent[i] = (bins[i] + bins[i+1])/2

    # return bin centers and luminosity function values
    return([logLCent, lFunc, vals, model])

# function to make a map given a set of parameters
def makeMapAndLumFunc(params, verbose=False, noise=0):
    llm.debug.verbose = verbose

    if llm.debug.verbose:
        llm.write_time('Starting Line Intensity Mapper')

    # Setup maps to output
    mapinst   = llm.params_to_mapinst(params)

    # Load halos from catalogue
    halos, cosmo = llm.load_peakpatch_catalogue(params.halo_catalogue_file, verbose=llm.debug.verbose)
    halos        = llm.cull_peakpatch_catalogue(halos, params.min_mass, mapinst, verbose=llm.debug.verbose)

    # Calculate Luminosity of each halo
    halos.Lco    = llm.Mhalo_to_Lco(halos, params.model, params.coeffs, verbose=llm.debug.verbose)

    # generate the luminosity function
    lumInfo = makeLumFunc(halos, params.model)

    # Bin halo luminosities into map
    mapinst.maps = llm.Lco_to_map(halos,mapinst, verbose=llm.debug.verbose, noise=noise)

    if llm.debug.verbose:
        llm.write_time('Finished Line Intensity Map Generation')

    return(mapinst, lumInfo)

# function to make and save a map given a set of parameters
def makeAndSaveMapAndLumFunc(params, verbose=False, noise=0):
    llm.debug.verbose = verbose

    if llm.debug.verbose:
        llm.write_time('Starting Line Intensity Mapper')

    # Setup maps to output
    mapinst   = llm.params_to_mapinst(params);

    # Load halos from catalogue
    halos, cosmo = llm.load_peakpatch_catalogue(params.halo_catalogue_file, verbose=llm.debug.verbose)
    halos        = llm.cull_peakpatch_catalogue(halos, params.min_mass, mapinst, verbose=llm.debug.verbose)

    # Calculate Luminosity of each halo
    halos.Lco    = llm.Mhalo_to_Lco(halos, params.model, params.coeffs, verbose=llm.debug.verbose)

    # generate the luminosity function
    lumInfo = makeLumFunc(halos, params.model)

    # Bin halo luminosities into map
    mapinst.maps = llm.Lco_to_map(halos,mapinst, verbose=llm.debug.verbose, noise=noise)

    if llm.debug.verbose:
        llm.write_time('Finished Line Intensity Map Generation')

    # save the map and lum function
    saveMapAndLum(mapinst, lumInfo)
    return()

# function to make maps and luminosity functions for every subfield-catalog in a directory that isn't already made
def makeCatdMaps(params, catalogList, catLoc, mapLoc, verbose=False):
    paramsDict = {}

    # get all of the base names of maps already made
    baseNames = loadBaseFNames(mapLoc)

    for cat in catalogList:

        # make sure a map hasn't already been bade
        if cat[:-4] in baseNames:
            if verbose:
                print('The map and luminosity function for {} have already been made\n\tMoving on to the next catalog'.format(cat))
            continue

        # set the base map params
        paramsDict['halo_catalogue_file'] = catLoc + '/' + cat
        paramsDict['map_output_file'] = mapLoc + '/' + cat
        params = setParams(paramsDict, params)

        # make the map and save it
        if verbose:
            print('\n\nAbout to generate map for catalog {}...'.format(cat))
        maps, lumInfo = makeMapAndLumFunc(params, verbose)
        saveMapAndLum(maps, lumInfo)

# modify the parameters object given a dict of the parametes
def setParams(paramsDict, params):
    for key, val in paramsDict.items():
        setattr(params, key, val)

    return(params)

# make a paramter object from a dict
class params(object):
    def __init__(self,d):
        self.__dict__ = d

# make a default parameter object
def defaultParams():
    paramsDict = {}
    catName = 'COMAP_z2.39-3.44_1140Mpc_seed_13579_rotate_0_subfield_0.npz'
    # get the map and catalog output file name
    paramsDict['halo_catalogue_file'] = '../catalogues/'+ catName
    paramsDict['map_output_file'] = '../maps/' + catName

    # halo luminosity function
    paramsDict['model'] = 'Li'
    paramsDict['coeffs'] = [0.0, 1.37, -1.74, 0.3, 0.3]
    paramsDict['min_mass'] = 2.5e10

    # map parameters
    paramsDict['nu_rest'] = 115.27
    paramsDict['nu_i'] = 34.
    paramsDict['nu_f'] = 26.
    paramsDict['nmaps'] = 100
    paramsDict['fov_x'] = 1.4
    paramsDict['fov_y'] = 1.4
    paramsDict['npix_x'] = 256
    paramsDict['npix_y'] = 256

    # Plot parameters
    paramsDict['plot_cube'] = True
    paramsDict['plot_pspec'] = True

    param = params(paramsDict)

    return(param)

# function to get non-default parameters
# model and coeffs have default values, but the other ones do not
def getParams(haloCat, mapFile, model='Li', coeffs=None, **kwargs):
    # set default params and ones from function variables
    params = defaultParams()
    params.halo_catalogue_file = haloCat
    params.map_output_file = mapFile
    params.model=model
    params.coeffs=coeffs

    # set params by keywords
    if kwargs is not None:
        for key, value in kwargs.items():
            if key in params.__dict__:
                params.__dict__[key] = value
            else:
                print('Field {} does not exist in the parameter object'.format(key))

    return(params)

# function to make a random map given a selection of catalogs, the catalog location and the location to store the map
def make_random_map(catalogs, haloLoc, mapLoc, default=False, noise=0,
    npix_x=None, npix_y=None, nmaps=None):
    # choose a random catalog
    catalog = random.choice(catalogs)
    # make the random map
    make_random_map_from_cat(catalog, haloLoc, mapLoc, default=default, noise=noise,
        npix_x=npix_x, npix_y=npix_y, nmaps=nmaps)

    return()

# function to make a random map given a specific catalog, catalog location and location to store the map
def make_random_map_from_cat(catalog, haloLoc, mapLoc, model=None, default=False,
    noise=0, npix_x=None, npix_y=None, nmaps=None):
    # make a random paramDict
    paramDict = {}
    paramDict = make_paramDict(paramDict=paramDict, model=model, default=default,
        npix_x=npix_x, npix_y=npix_y, nmaps=nmaps)

    # make sure the map directory exists
    checkDirectoryPath(mapLoc)

    # set the parameters for limlam_mocker
    param = getParams(haloLoc + catalog, mapLoc + catalog, **paramDict)

    # get a file name showing the info about model and coeffs used
    file_name = coeffs_to_file_name(param.model, param.coeffs, mapLoc, catalog)
    param.map_output_file = file_name
    # print(param.halo_catalogue_file, '\n', param.map_output_file, '\n', param.model, '\n', param.coeffs)
    makeAndSaveMapAndLumFunc(param, verbose=False, noise=noise)

    return()

# function to make a random paramDict for a limlam_mocker run
# randomizes the model and the values of the coefficients in the model if requested
def make_paramDict(paramDict, model=None, default=False,
    npix_x=None, npix_y=None, nmaps=None):
    # list of models
    # model_list = ['Li', 'Padmanabhan', 'Breysse']
    model_list = ['Li', 'Padmanabhan']
    means = []
    sig = []

    # randomly pick a model if none is given
    if model == None:
        model = random.choice(model_list)

    # set up the coefficients for the model
    if model == 'Li':
        # parameter info from arxiv:1503.08833
        # log_delta_mf, alpha, beta, sigma_sfr (>0), sigma_lc0 (>0)
        means = [0.0, 1.37,-1.74, 0.3, 0.3]
        # loose priors
        # sig = [0.3, 0.37, 3.74, 0.1, 0.1]
        sig = [0.03, 0.037, .374, 0.01, 0.01]
        # stronger priors
        #sig = [0.3, 0.04, 0.4, 0.1, 0.1]
    elif model == 'Padmanabhan':
        # parameter info from arxiv:1706.01471
        # m10, m11, n10, n11, b10, b11, y10, y11
        means = [4.17e12, -1.17, 0.0033, 0.04, 0.95, 0.48, 0.66, -0.33]
        # priors
        sig = [2.03e12, 0.85, 0.0016, 0.03, 0.46, 0.35, 0.32, 0.24]
    elif model == 'Breysse':
        # parameter info from arxiv:1706.01471
        # A, b
        means = [2e-6, 1.0]
        # priors
        sig = [5e-7, 0.125]
    else:
        sys.exit('\n\n\tYour model, '+model+', does not seem to exist\n\t\tPlease check src/halos_to_luminosity.py in limlam_mocker to add it\n\n')

    # set the model parameter for limlam
    paramDict['model'] = model

    # if default is set, use the default values
    if default:
        coeffs = means
    # if default is not set, find a random realization of the given model's parameters
    else:
        coeffs = [np.random.normal(x,y) for x,y in zip(means, sig)]

        # make sure last two parameters in the Li model are non-negative
        # as well as the alpha parameter
        if model == 'Li':
            for i in [1, -1,-2]:
                if coeffs[i] <= 0:
                    coeffs[i] = 1e-20
        if model == 'Padmanabhan':
            for i in [0]:
                if coeffs[i] <= 0:
                    coeffs[i] = 1e-20

    # set coeffs parameter for limlam
    paramDict['coeffs'] = coeffs

    # see about changing the default number of pixels or maps used
    if npix_x is not None:
        paramDict['npix_x'] = npix_x
    if npix_y is not None:
        paramDict['npix_y'] = npix_y
    if nmaps is not None:
        paramDict['nmaps'] = nmaps

    return(paramDict)

# function to convert model and catalog info into a map filename
# filenames are given catalog info __ model _ coeffs
def coeffs_to_file_name(model, coeffs, mapLoc, catalog):
    file_name = '__' + model + '_'

    # put coeffs into the filename
    for coeff in coeffs:
        file_name += '{:.3e}_'.format(coeff)

    # form the full filename
    full_file_name = mapLoc + catalog[:-4] + file_name[:-1] + catalog[-4:]

    return(full_file_name)

# function to convert a path to a map into info about the map
def path_to_coeffs(path):
    # get the filename
    file_name = path.split('/')[-1]
    # get info about the filename
    model, coeffs = file_name_to_coeffs(file_name)

    return(model, coeffs)

# function to convert a filename for a map into info about the map
def file_name_to_coeffs(file_name):
    # split catalog and model info
    cat, lum_info = file_name.split('__')

    # split model info up and get the specific model as well as coeffs used
    lum_list = lum_info[:-4].split('_')
    model = lum_list[0]
    coeffs = list(map(float, lum_list[1:]))

    return(model, coeffs)
