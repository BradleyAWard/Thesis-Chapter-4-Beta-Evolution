# --------------------------------------------------
# Imports
# --------------------------------------------------

import numpy as np
from astropy.constants import c, h, k_B, M_sun
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo

LAM_UM_REF = 850
KAPPA_REF = 0.077

# --------------------------------------------------
# Modified Blackbody Functions
# --------------------------------------------------

def blackbody(nu, t):
    """
    Blackbody function
    
    :param nu: Rest frame frequency [Hz]
    :param t: Dust temperature [K]
    :return: Planck function with dust temperature T
    """
    nu = nu*u.Hz
    t = t*u.K
    bb = ((2*h*(nu**3))/(c**2))*(1/(np.exp((h*nu)/(k_B*t))-1))
    bb = bb.to(u.Jy)
    return bb

# --------------------------------------------------

def get_kappa(nu, beta, lam_um_ref=LAM_UM_REF, kappa_ref=KAPPA_REF):
    """
    Returns the dust absorption coefficient
    
    :param nu: Rest frame frequency [Hz]
    :param beta: Dust emissivity spectral index
    :param lam_um_ref: Rest frame reference wavelength [microns]
    :param kappa_ref: Reference dust absorption coefficient [m2 kg-1]
    :return: Dust absorption coefficient [m2 kg-1]
    """
    nu = nu * u.Hz
    lam_ref = (lam_um_ref * u.micron).to(u.m)
    nu_ref = c / lam_ref
    kappa_ref = kappa_ref * ((u.m ** 2) / u.kg)
    kappa = kappa_ref * ((nu / nu_ref) ** beta)
    return kappa

# --------------------------------------------------

def get_tau(nu, lam_um_thick, beta):
    """
    Returns the optical depth
    
    :param nu: Rest frame frequency [Hz]
    :param lam_um_thick: Wavelength for optically thin/thick border [microns]
    :param beta: Dust emissivity spectral index
    :return: Optical depth
    """
    nu = nu * u.Hz
    lam_thick = (lam_um_thick * u.micron).to(u.m)
    nu_thick = c / lam_thick
    tau = (nu/nu_thick)**beta
    return tau

# --------------------------------------------------

def find_cutoff_frequency(nu, mbb, alpha):
    """
    Returns the mid-IR/far-IR transition wavelength
    
    :param nu: Rest frame frequency [Hz]
    :param mbb: Modified blackbody function
    :param alpha: Mid-IR power law
    :return: Cutoff frequency [Hz]
    """
    log_nu = np.log10(nu)
    log_bb = np.log10(mbb)
    del_x = np.diff(log_nu)
    del_y = np.diff(log_bb)
    grad = del_y / del_x
    nu_c = nu[np.searchsorted(grad, -alpha)]
    return nu_c

# --------------------------------------------------

def add_powerlaw(nu, mbb, nu_c, mbb_nu_c, alpha):
    """
    Adds a power law function to a model
    
    :param nu: Rest frame frequency [Hz]
    :param mbb: Modified blackbody function
    :param nu_c: Cutoff mid-IR/far-IR wavelength [microns]
    :param mbb_nu_c: Blackbody function at cutoff wavelength
    :param alpha: Mid-IR power law
    :return: Power law function
    """
    n_pl = mbb_nu_c * (nu_c ** alpha)
    powerlaw = nu ** (-alpha)
    pl = n_pl * powerlaw
    mbb_pl = np.where(nu <= nu_c, mbb, pl)
    return mbb_pl

# --------------------------------------------------
# CMB Contribution
# --------------------------------------------------

def cmb_heating(z, t0, beta):
    """
    Term for heating of ISM from CMB photons
    
    :param z: Redshift
    :param t0: Temperature at redshift 0 [K]
    :param beta: Dust emissivity spectral index
    :return: Temperature at redshift z [K]
    """
    return ((t0**(4+beta))+((2.725**(4+beta))*(((1+z)**(4+beta))-1)))**(1/(4+beta))

# --------------------------------------------------

def f_cmb(nu, t, z):
    """
    Term for fraction of SED observed against CMB
    
    :param nu: Rest frame frequency [Hz] 
    :param t: Dust temperature [K]
    :param z: Redshift
    :return: Fraction of SED observed against CMB
    """
    t_cmb = 2.725*(1+z)
    return 1-(blackbody(nu, t_cmb)/blackbody(nu, t))

# --------------------------------------------------
# Optically Thin Model
# --------------------------------------------------

def modified_blackbody_optically_thin(nu, z, theta, m=1, cmb=True):
    """
    Optically thin modified blackbody model
    
    :param nu: Rest frame frequency [Hz]
    :param z: Redshift
    :param theta: Fitting parameters
    :param m: Lensing magnification (Default = 1)
    :param cmb: Inclusion of CMB effects (Default = TRUE)
    :return: Optically thin MBB model
    """
    log_m, t, beta = theta

    # Include effects of the CMB
    if cmb:
        t_z = cmb_heating(z, t, beta)
        t = t_z

    mass = (10 ** log_m) * M_sun
    d = cosmo.luminosity_distance(z=z).to(u.m)/(1+z)
    bb = blackbody(nu, t)

    kappa = get_kappa(nu, beta)

    sigma_times_a = mass
    tau_times_a = sigma_times_a * kappa
    omega_divided_a = 1 / (d ** 2)

    omega = omega_divided_a
    modifier = tau_times_a

    mbb_rest = omega * m * modifier * bb
    if cmb:
        mbb_rest = mbb_rest * f_cmb(nu, t, z)

    mbb_obs = mbb_rest/(1+z)
    return mbb_obs.value