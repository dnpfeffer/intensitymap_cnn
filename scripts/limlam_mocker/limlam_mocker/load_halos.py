from __future__ import absolute_import, print_function
import numpy as np
from  .tools import *
from . import debug

@timeme
def load_peakpatch_catalogue(filein, **kw):
    """
    Load peak patch halo catalogue into halos class and cosmology into cosmo class

    Returns
    -------
    halos : class
        Contains all halo information (position, redshift, etc..)
    cosmo : class
        Contains all cosmology information (Omega_i, sigme_8, etc)
    """
    halos      = empty_table()            # creates empty class to put any halo info into
    cosmo      = empty_table()            # creates empty class to put any cosmology info into

    halo_info  = np.load(filein)
    if debug.verbose: print("\thalo catalogue contains:\n\t\t", halo_info.files)

    #get cosmology from halo catalogue
    params_dict    = halo_info['cosmo_header'][()]
    cosmo.Omega_M  = params_dict.get('Omega_M')
    cosmo.Omega_B  = params_dict.get('Omega_B')
    cosmo.Omega_L  = params_dict.get('Omega_L')
    cosmo.h        = params_dict.get('h'      )
    cosmo.ns       = params_dict.get('ns'     )
    cosmo.sigma8   = params_dict.get('sigma8' )

    cen_x_fov  = params_dict.get('cen_x_fov', 0.) #if the halo catalogue is not centered along the z axis
    cen_y_fov  = params_dict.get('cen_y_fov', 0.) #if the halo catalogue is not centered along the z axis

    halos.M          = halo_info['M']     # halo mass in Msun
    halos.x_pos      = halo_info['x']     # halo x position in comoving Mpc
    halos.y_pos      = halo_info['y']     # halo y position in comoving Mpc
    halos.z_pos      = halo_info['z']     # halo z position in comoving Mpc
    halos.vx         = halo_info['vx']    # halo x velocity in km/s
    halos.vy         = halo_info['vy']    # halo y velocity in km/s
    halos.vz         = halo_info['vz']    # halo z velocity in km/s
    halos.redshift   = halo_info['zhalo'] # observed redshift incl velocities
    halos.zformation = halo_info['zform'] # formation redshift of halo

    halos.nhalo = len(halos.M)

    halos.chi        = np.sqrt(halos.x_pos**2+halos.y_pos**2+halos.z_pos**2)
    halos.ra         = np.arctan2(-halos.x_pos,halos.z_pos)*180./np.pi - cen_x_fov
    halos.dec        = np.arcsin(  halos.y_pos/halos.chi  )*180./np.pi - cen_y_fov

    assert np.max(halos.M) < 1.e17,             "Halos seem too massive"
    assert np.max(halos.redshift) < 4.,         "need to change max redshift interpolation in tools.py"
    assert (cosmo.Omega_M + cosmo.Omega_L)==1., "Does not seem to be flat universe cosmology"

    if debug.verbose: print('\n\t%d halos loaded' % halos.nhalo)

    return halos, cosmo

@timeme
def cull_peakpatch_catalogue(halos, min_mass, mapinst, **kw):
    """
    crops the halo catalogue to only include desired halos
    """
    dm = [(halos.M > min_mass) * (halos.redshift >= mapinst.z_i)
                               * (np.abs(halos.ra) <= mapinst.fov_x/2)
                               * (np.abs(halos.dec) <= mapinst.fov_y/2)
                               * (halos.redshift <= mapinst.z_f)]

    for i in dir(halos):
        if i[0]=='_': continue
        try:
            setattr(halos,i,getattr(halos,i)[tuple(dm)])
        except TypeError:
            pass
    halos.nhalo = len(halos.M)

    if debug.verbose: print('\n\t%d halos remain after mass/map cut' % halos.nhalo)

    return halos
