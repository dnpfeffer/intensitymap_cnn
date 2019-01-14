import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.interpolate
import sys
import os
from . import debug
from .tools import *

sfr_interp_tab = None

@timeme
def Mhalo_to_Lco(halos, model, coeffs, **kw):
    """
    General function to get L_co(M_halo) given a certain model <model>
    if adding your own model follow this structure,
    and simply specify the model to use in the parameter file

    Parameters
    ----------
    halos : class
        Contains all halo information (position, redshift, etc..)
    model : str
        Model to use, specified in the parameter file
    coeffs :
        None for default coeffs
    """
    dict = {'Li':          Mhalo_to_Lco_Li,
            'Padmanabhan': Mhalo_to_Lco_Padmanabhan,
            'Breysse':     Mhalo_to_Lco_Breysse}

    if model in dict.keys():
        return dict[model](halos, coeffs)

    else:
        sys.exit('\n\n\tYour model, '+model+', does not seem to exist\n\t\tPlease check src/halos_to_luminosity.py to add it\n\n')


def Mhalo_to_Lco_Li(halos, coeffs):
    """
    halo mass to SFR to L_CO
    following the Tony li 2016 model
    arXiv 1503.08833
    """
    global sfr_interp_tab
    if coeffs is None:
        # Power law parameters from paper
        log_delta_mf,alpha,beta,sigma_sfr,sigma_lco = (
            0.0, 1.37,-1.74, 0.3, 0.3)
    else:
        log_delta_mf,alpha,beta,sigma_sfr,sigma_lco = coeffs;
    delta_mf = 10**log_delta_mf;

    # Get Star formation rate
    if sfr_interp_tab is None:
        sfr_interp_tab = get_sfr_table()
    sfr            = sfr_interp_tab.ev(np.log10(halos.M), np.log10(halos.redshift+1))
    sfr            = add_log_normal_scatter(sfr, sigma_sfr)

    # infrared luminosity
    lir      = sfr * 1e10 / delta_mf
    alphainv = 1./alpha
    # Lco' (observers units)
    try:
        Lcop     = lir**alphainv * 10**(-beta * alphainv)
    except e:
        print(lir, alphainv, beta)
        exit(0)
    # Lco in L_sun
    Lco      =  4.9e-5 * Lcop
    Lco      = add_log_normal_scatter(Lco, sigma_lco)

    if debug.verbose: print('\n\tMhalo to Lco calculated')

    return Lco

def Mhalo_to_Lco_Padmanabhan(halos, coeffs):
    """
    halo mass to L_CO
    following the Padmanabhan 2017 model
    arXiv 1706.01471
    """
    if coeffs is None:
        m10,m11,n10,n11,b10,b11,y10,y11 = (
            4.17e12,-1.17,0.0033,0.04,0.95,0.48,0.66,-0.33)
    else:
        m10,m11,n10,n11,b10,b11,y10,y11 = coeffs

    z  = halos.redshift
    hm = halos.M

    m1 = 10**(np.log10(m10)+m11*z/(z+1))
    n  = n10 + n11 * z/(z+1)
    b  = b10 + b11 * z/(z+1)
    y  = y10 + y11 * z/(z+1)

    Lprime = 2 * n * hm / ( (hm/m1)**(-b) + (hm/m1)**y )
    Lco    = 4.9e-5 * Lprime

    return Lco

def get_sfr_table():
    """
    Load SFR Table
    Columns are: z+1, logmass, logsfr, logstellarmass
    Intermediate processing of tabulated data
    """

    tablepath = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    tablepath+= '/tables/sfr_behroozi_release.dat'
    dat_zp1, dat_logm, dat_logsfr, _ = np.loadtxt(tablepath, unpack=True)

    dat_logzp1 = np.log10(dat_zp1)
    dat_sfr    = 10.**dat_logsfr

    # Reshape arrays
    dat_logzp1  = np.unique(dat_logzp1)    # log(z), 1D
    dat_logm    = np.unique(dat_logm)    # log(Mhalo), 1D
    dat_sfr     = np.reshape(dat_sfr, (dat_logm.size, dat_logzp1.size))

    # Get interpolated SFR value(s)
    sfr_interp_tab = sp.interpolate.RectBivariateSpline(dat_logm, dat_logzp1, dat_sfr,
                                                        kx=1, ky=1)
    return sfr_interp_tab

def Mhalo_to_Lco_Breysse(halos, coeffs):
    """
    halo mass to SFR to L_CO
    following the Breysse et al 2015 model
    arXiv 1503.05202
    """
    # A varies between 6.24e-7 and 9.6e-6 (arxiv:1405.0489), b varies between 0.8 and 1.2 (Direct from Pat)
    if coeffs is None:
        A,b = (2e-6,1.0)
    else:
        A,b = coeffs

    hm = halos.M

    Lco = A * hm**b

    return Lco

def add_log_normal_scatter(data,dex):
    """
    Return array x, randomly scattered by a log-normal distribution with sigma=dexscatter.
    [via @tonyyli - https://github.com/dongwooc/imapper2]
    Note: scatter maintains mean in linear space (not log space).
    """
    if (dex<=0):
        return data
    # Calculate random scalings
    sigma       = dex * 2.302585 # Stdev in log space (DIFFERENT from stdev in linear space), note: ln(10)=2.302585
    mu          = -0.5*sigma**2

    randscaling = np.random.lognormal(mu, sigma, data.shape)
    xscattered  = np.where(data > 0, data*randscaling, data)

    return xscattered
