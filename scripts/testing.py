from __future__ import division
import numpy              as np
import matplotlib.pylab   as plt
import scipy              as sp
from limlam_mocker import limlam_mocker as llm
#Get Parameters for run
from limlam_mocker import params        as params

import lnn                as lnn



paramsDict = {}
paramsDict['halo_catalogue_file'] = '../limlam_mocker/'+ params.halo_catalogue_file
paramsDict['map_output_file'] = '../maps/Lco_cube_newer.npz'

params = lnn.setParams(paramsDict, params)

# params.halo_catalogue_file = '../limlam_mocker/'+ params.halo_catalogue_file
# makeAndSaveMap(params, verbose=False)

# maps, lumInfo = lnn.makeMapAndLumFunc(params)
# lnn.saveMapAndLum(maps, lumInfo)

catLoc = '../catalogues'
mapLoc = '../maps'
subFieldCats = lnn.loadSubFields(catLoc)
print(subFieldCats)
lnn.makeCatdMaps(params, subFieldCats, catLoc, mapLoc, verbose=True)

