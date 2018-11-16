import numpy as np

from limlam_mocker import limlam_mocker as llm
import lnn as lnn

### set the location of the halo catalogs and maps
haloLoc = '../catalogues/'
mapLoc = '../maps/test/'

### set the halo luminosity information
model = 'Li'
coeffs = None
#min_mass = 2.5e10

### set noise
noise = 0

### get the subfield names
subFields = lnn.loadSubFields(haloLoc)

### set the paramDict
paramDict = {}
paramDict['model'] = model
paramDict['coeffs'] = coeffs

### make sure the map directory exists
lnn.checkDirectoryPath(mapLoc)

for sub in subFields:
    #if 'COMAP_z2.39-3.44_1140Mpc_seed_13649_rotate_0_subfield_0.npz' in sub:
    param = lnn.getParams(haloLoc + sub, mapLoc + sub, **paramDict)
    lnn.makeAndSaveMapAndLumFunc(param, verbose=False, noise=noise)


