import numpy as np

from limlam_mocker import limlam_mocker as llm
import lnn as lnn

### set the location of the halo catalogs and maps
haloLoc = '../catalogues2/'
mapLoc = '../maps2/random_small_Li_small_map/'

### what model to use and if the maps should use random values or not
model = 'Li'
default = False

### what level of noise to use
noise = 0

### make sure the map directory exists
lnn.checkDirectoryPath(mapLoc)

### get the subfield names
subFields = lnn.loadSubFields(haloLoc)

### number of maps to make
numb_maps = len(subFields)

### start making random maps
for i in range(numb_maps):
<<<<<<< HEAD
    lnn.make_random_map_from_cat(subFields[i], haloLoc, mapLoc, model, default, noise=noise)
=======
    lnn.make_random_map_from_cat(subFields[i], haloLoc, mapLoc, model, default, noise=noise,
                    npix_x=npix_x, npix_y=npix_y, nmaps=nmaps)
>>>>>>> 934670f2083f421aa82704b8e08184493bbf734c

    ### print every 100 maps where we are in the process
    if i%100 == 0:
            print('Finished making map {} out of {} ({:.3f}%)'.format(i, numb_maps, i/numb_maps*100))
