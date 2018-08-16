### stolen from George Stein's limlam_mocker
### modified by Daniel Pfeffer

from __future__ import absolute_import, print_function
import numpy as np

import sys
from os import listdir
from os.path import isfile, join

class empty_table():
    def __init__(self):
        pass
    def copy(self):
        return copy.copy(self)

def load_peakpatch_catalogue(filein, rotate_theta=0, verbose=False):
    halos      = empty_table()            # creates empty class to put any halo info into
    cosmo      = empty_table()            # creates empty class to put any cosmology info into

    halo_info  = np.load(filein)
    if verbose:
        print("\n\thalo catalogue contains:\n\t\t", halo_info.files)

    #get cosmology from halo catalogue
    params_dict    = halo_info['cosmo_header'][()]
    cosmo.Omega_M  = params_dict.get('Omega_M')
    cosmo.Omega_B  = params_dict.get('Omega_B')
    cosmo.Omega_L  = params_dict.get('Omega_L')
    cosmo.h        = params_dict.get('h'      )
    cosmo.ns       = params_dict.get('ns'     )
    cosmo.sigma8   = params_dict.get('sigma8' )

    halos.M          = halo_info['M']     # halo mass in Msun

    ### rotate the catalog if a rotation angle is given
    if rotate_theta != 0:
        if verbose:
            print("\n\trotate catalogs by {} degrees".format(rotate_theta))
        m = [[np.cos(rotate_theta*np.pi/180), -np.sin(rotate_theta*np.pi/180)], [np.sin(rotate_theta*np.pi/180), np.cos(rotate_theta*np.pi/180)]]
        halos.x_pos, halos.y_pos  = np.matmul(m, [halo_info['x'], halo_info['y']])
    else:
        halos.x_pos      = halo_info['x']     # halo position in comoving Mpc
        halos.y_pos      = halo_info['y']
    halos.z_pos      = halo_info['z']
    halos.chi        = np.sqrt(halos.x_pos**2+halos.y_pos**2+halos.z_pos**2)

    halos.vx         = halo_info['vx']    # halo velocity in km/s
    halos.vy         = halo_info['vy']
    halos.vz         = halo_info['vz']

    halos.redshift   = halo_info['zhalo'] # observed redshift incl velocities

    halos.zformation = halo_info['zform'] # formation redshift of halo

    halos.nhalo = len(halos.M)

    halos.ra         = np.arctan2(-halos.x_pos,halos.z_pos)*180./np.pi
    halos.dec        = np.arcsin(  halos.y_pos/halos.chi  )*180./np.pi

    if verbose:
        print('\n\t%d halos loaded' % halos.nhalo)

    return halos, cosmo

def cull_peakpatch_catalogue(halos,fov_x_l, fov_x_r, fov_y_l, fov_y_r, verbose=False):
    halosi      = empty_table()            # creates empty class to put any halo info into

    dm = [(halos.ra > fov_x_l)
          * (halos.ra < fov_x_r)
          * (halos.dec > fov_y_l)
          * (halos.dec < fov_y_r)]


    for i in dir(halos):
        if i[0]=='_': continue
        try:
            setattr(halosi,i,getattr(halos,i)[tuple(dm)])
        except TypeError:
            pass
    halosi.nhalo = len(halosi.M)

    if verbose:
        print('\n\t%d halos remain after map cut' % halosi.nhalo)

    return halosi

def split_catalogue(filein, rotate_theta=0, verbose=False):
    ### the normal fox of the catalog is something like 11.52 degrees, but the number of halos drops off on the edges
    ### the 9.52 makes sure the edge effects are ignored
    # fov_x  = 9.52
    # fov_y  = 9.52

    ### debug fox size
    fov_x  = 3.0
    fov_y  = 3.0

    fov_x_subfield = 1.4
    fov_y_subfield = 1.4

    n_subfield   = int((fov_x//fov_x_subfield) * (fov_y//fov_y_subfield))
    n_y_subfield = int(fov_x//fov_x_subfield)
    n_x_subfield = int(fov_y//fov_y_subfield)

    fov_x_new  = n_x_subfield*fov_x_subfield
    fov_y_new  = n_y_subfield*fov_y_subfield

    if verbose:
        bar = 90*'-'
        print('\n'+bar+'\n',n_subfield,' subfields of',fov_x_subfield,'deg x',fov_y_subfield,'deg can fit into the full',fov_x,'deg x',fov_y,'deg fov','\n'+bar+'\n',)

    # Load in full halo catalogue
    halos, cosmo = load_peakpatch_catalogue(filein, rotate_theta, verbose)

    # loop over each subfield and save halos as seperate files
    for i in range(n_y_subfield):
        for j in range(n_x_subfield):

            fov_x_l  = -fov_x_new/2 + fov_x_subfield * (j  )
            fov_x_r  = -fov_x_new/2 + fov_x_subfield * (j+1)

            fov_y_l  = -fov_y_new/2 + fov_y_subfield * (i  )
            fov_y_r  = -fov_y_new/2 + fov_y_subfield * (i+1)

            cen_fov_x = -fov_x_new/2 + fov_x_subfield * (j+1./2)
            cen_fov_y = -fov_y_new/2 + fov_y_subfield * (i+1./2)

            # cut halos outside sub-field of view
            halosi = cull_peakpatch_catalogue(halos, fov_x_l, fov_x_r, fov_y_l, fov_y_r, verbose)

            ifileout = i*n_x_subfield+j
            print("writing subfield: ", ifileout)

            ### add in rotation info to the filename
            rotate_str = '_rotate_{0:d}'.format(int(rotate_theta))

            fileout = filein[:-4]+rotate_str+'_subfield_'+str(ifileout)

            # save to file, with cen_x_fov, cen_y_fov denoting the center of the field of view kept
            cosmo_header   = {'Omega_M': cosmo.Omega_M, 'Omega_B': cosmo.Omega_B, 'Omega_L': cosmo.Omega_L,
                              'h':cosmo.h, 'ns':cosmo.ns, 'sigma8':cosmo.sigma8,
                              'cen_x_fov':cen_fov_x, 'cen_y_fov':cen_fov_y}
            np.savez(fileout, cosmo_header=cosmo_header,
                     x=halosi.x_pos, y=halosi.y_pos, z=halosi.z_pos,
                     vx=halosi.vx, vy=halosi.vy, vz=halosi.vz,
                     M=halosi.M, zhalo=halosi.redshift,zform=halosi.zformation)




### if run as an individual file then go through each non python, non subfield and non hidden file and make subfields for the remaining files
def main(rotate_theta=0):
    ### get non subfield catalogs
    onlyfiles = [f for f in listdir('../catalogues') if (isfile(join('./', f)) and f[0] != '.' and '.py' not in f and 'subfield' not in f)]

    for f in onlyfiles:
        split_catalogue(f, rotate_theta, verbose=True)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(int(sys.argv[1]))
    else:
        main()
