import numpy as np

from limlam_mocker import limlam_mocker as llm
import lnn as lnn

### set the location of the halo catalogs and maps
haloLoc = '../catalogues2/'
# mapLoc = '../maps2/random_maps_small_map/'
mapLoc = '../maps2/test/'

# size of LIM to use
npix_x = 64
npix_y = 64
nmaps = 10

### what noise level to use
noise = 0

### to use base coefficients for the model or not
default = True

### make sure the map directory exists
lnn.checkDirectoryPath(mapLoc)

### get the subfield names
subFields = lnn.loadSubFields(haloLoc)

### number of maps to make
numb_maps = 5796

### start making random maps
for i in range(numb_maps):
    lnn.make_random_map(subFields, haloLoc, mapLoc, default, noise=noise,
                    npix_x=npix_x, npix_y=npix_y, nmaps=nmaps)

    ### print every 100 maps where we are in the process
    if i%100 == 0:
            print('Finished making map {} out of {} ({:.3f}%)'.format(i, numb_maps, i/numb_maps*100))
