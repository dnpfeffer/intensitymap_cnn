import numpy as np

# needed to save the map
from limlam_mocker import limlam_mocker as llm

# used for file handeling
from pathlib import Path
import os
import pickle

# make and save a map given a set of parameters
def saveMapAndLum(maps, lumInfo):
    # Output map to file
    fileName = Path(maps.output_file)
    loc = fileName.parent
    name = fileName.name[:-len(fileName.suffix)] + '_map.npz'
    saveMap(maps, loc=loc, name=name)

    # modify the name for the luminosity function
    name = fileName.name[:-len(fileName.suffix)] + '_lum.npz'
    saveLumFunc(lumInfo, loc=loc, name=name)

# wrapper function to save a map
# has function to change the name and location of the file
def saveMap(maps, loc='.', name='test_map.npz'):
    # make a new file name with the correct path
    fileName = Path('{0}/{1}'.format(loc, name))
    maps.output_file = str(fileName)

    # save the file
    llm.save_maps(maps, verbose=llm.debug.verbose)

    return()

# function to save the luminosity function
def saveLumFunc(lumInfo, loc='.', name='test_lum.npz'):
    # make a new file name with the correct path
    fileName = Path('{0}/{1}'.format(loc, name))

    # save the file
    np.savez(str(fileName),
             logBinCent=lumInfo[0],
             numberCt=lumInfo[1],
             lumFunc=lumInfo[2],
             model=lumInfo[3])

    return()

# function to make sure the path to a given location exists
def checkDirectoryPath(loc):
    path = Path(loc)

    if os.path.isdir(loc) == False:
        path.mkdir(parents=True)

    return()

# function to load any tpye of data np can load
def loadData(fName):
    loaded_data = np.load(fName)
    return(loaded_data)

# function to load map data that isn't the map_cube
def loadMap_data(fName):
    info = {}
    with np.load(fName) as data:
        info['omega_pix'] = data['pix_size_x'] * data['pix_size_y']
        info['nu'] = data['map_frequencies']

    return(info)

# function to load an intensity map
def loadMap(fName):
    with np.load(fName) as data:
        mapData = data['map_cube']

    return(mapData)

# function to load an intensity map
def loadLums(fName, lumByproduct='basic'):
    with np.load(fName) as data:
        lumData = lumFuncByproduct(data, lumByproduct=lumByproduct)

    return(lumData)

# function to load the map and lum function for a given base file name
def loadMapAndLum(baseFName, lumByproduct='basic'):
    mapData = loadMap(baseFName + '_map.npz')
    lumData = loadLums(baseFName + '_lum.npz', lumByproduct=lumByproduct)

    return(mapData, lumData)

# function to directly load the log bin centers from a given base file name
def loadLogBinCenters(baseFName):
    lumLogBinCents = loadData(baseFName + '_lum.npz')['logBinCent']
    return(lumLogBinCents)

# function to get all of the subfield catalogs in a directory
def loadSubFields(loc):
    names = loadDirNpzs(loc)

    subFieldNames = [n for n in names if 'subfield' in n]

    return(subFieldNames)

# function to remove duplicates from a list
# the returned list will always be in the same order
def remove_duplicates(list):
    seen = set()
    seen_add = seen.add
    list = [x for x in list if not (x in seen or seen_add(x))]
    return(list)

# function to get all the base file names in a directory
def loadBaseFNames(loc):
    names = loadDirNpzs(loc)
    # -8 because _map and _lum have 4 characters and so does .npz
    baseNames = [n[:-8] for n in names]

    # remove duplicates
    baseNames = remove_duplicates(baseNames)
    return(baseNames)

# function to load all npz files in a direcotry
def loadDirNpzs(loc):
    path = Path(loc)
    names = []

    # iterate over each item in a directory and store the base names
    # for p in sorted(list(path.iterdir())):
    for p in path.iterdir():
        # ignore files that begin with '.'
        # ignore directories
        if p.name[0] == '.' or '/' in p.name:
            continue
        # only get .npz files
        if '.npz' not in p.name:
            continue


        # the 4 is because both _map and _lum have 4 characters
        names.append(p.name)

    # remove duplicates
    names = remove_duplicates(names)
    return(names)

# function to get the names of files in the model folder that
# match the given name
def get_model_name_matches(modelLoc, model_name):
    model_matches = []
    path = Path(modelLoc)
    for p in path.iterdir():
        # ignore files that begin with '.'
        # ignore directories
        if p.name[0] == '.' or '/' in p.name:
            continue
        # keep track of files that contain the correct name
        if model_name in p.name:
            model_matches.append(p.name)

    # return the names
    return(model_matches)

# function to load up the history of a model
def load_history(history_path):
    with open(history_path, 'rb') as pickle_file:
        history = pickle.load(pickle_file)
    return(history)

# function to save pickle objects
def save_pickle(data, file_name, file_type='.pkl'):
    with open(file_name + file_type, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    return()

# function to load pickle objects
def load_pickle(file_name, file_type='.pkl'):
    with open(file_name + file_type, 'rb') as f:
        data =  pickle.load(f)
    return(data)

# gets the number of times a model has been trained
def get_model_iteration(model_name, model_matches=[], model_loc=[]):
    # handle the possible inputs
    # can be given a list of names or the directory with the model files
    if len(model_matches) > 0 and len(model_loc) > 0:
        print('Given both a modelLoc and model match.  Defaulting to model match')
    elif len(model_loc) > 0 and len(model_matches) == 0:
        model_matches = get_model_name_matches(model_loc, model_name)

    # yell at the user if there was no real models with that name
    if len(model_matches) == 0 and len(model_loc) == 0:
        print('Either no model locations or model matches were given or there was not a completed run with the given model name')
        return(0)

    # get the number of models with that name with completed histories
    ct = 0
    for m in model_matches:
        if model_name + '_history' in m and m[:len(model_name)] == model_name:
            ct += 1

    ### return the count
    return(ct)

# gets the total history of a model
def get_full_history(model_name, model_loc):
    # get the nubmer of times the model has been trained
    train_count = get_model_iteration(model_name, model_loc=model_loc)

    # load up the first trianing history
    history_name = model_loc + model_name + '_history'
    base_history = load_history(history_name)

    # return the history if it was only trained once
    if train_count == 1:
        return(base_history)

    # combine histories if the model was trained multiple times
    for i in range(1, train_count):
        new_history = load_history('{0}_{1}'.format(history_name, i))
        for key in base_history:
            base_history[key] = base_history[key] + new_history[key]

    return(base_history)

# function to return the required luminosity function byproduct
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

# functiont hat converts a luminosity model into an int for categorizing
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
