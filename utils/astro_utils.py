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
    # Uncouple parameters
    log_m, t, beta = theta

    # Include effects of the CMB
    if cmb:
        t_z = cmb_heating(z, t, beta)
        t = t_z

    # Retrieve dust mass, distance, blackbody and dust absorption coefficient
    mass = (10 ** log_m) * M_sun
    d = cosmo.luminosity_distance(z=z).to(u.m)/(1+z)
    bb = blackbody(nu, t)
    kappa = get_kappa(nu, beta)

    # Normalization from dust mass, distance and dust absorption coefficient
    sigma_times_a = mass
    tau_times_a = sigma_times_a * kappa
    omega_divided_a = 1 / (d ** 2)
    omega = omega_divided_a
    modifier = tau_times_a

    # Combine blackbody with normalization, modifiers and CMB
    mbb_rest = omega * m * modifier * bb
    if cmb:
        mbb_rest = mbb_rest * f_cmb(nu, t, z)

    mbb_obs = mbb_rest/(1+z)
    return mbb_obs.value

# --------------------------------------------------

def modified_blackbody_optically_thin_powerlaw(nu, z, theta, m=1, cmb=True):
    """
    Optically thin modified balckbody model with mid-IR power law
    
    :param nu: Rest frame frequency [Hz]
    :param z: Redshift
    :param theta: Fitting parameters
    :param m: Lensing magnification (Default = 1)
    :param cmb: Inclusion of CMB effects (Default = TRUE)
    :return: Optically thin MBB model with mid-IR power law
    """
    # Uncouple parameters
    log_m, t, beta, alpha = theta
    theta_mbb = log_m, t, beta

    # Retrieve MBB model, find cutoff frequency, combine MBB with power law
    mbb = modified_blackbody_optically_thin(nu, z, theta_mbb, m, cmb)
    nu_c = find_cutoff_frequency(nu, mbb, alpha)
    mbb_nu_c = modified_blackbody_optically_thin(nu_c, z, theta_mbb, m, cmb)
    mbb_pl = add_powerlaw(nu, mbb, nu_c, mbb_nu_c, alpha)
    return mbb_pl

# --------------------------------------------------
# General Opacity Model (Continuum Area)
# --------------------------------------------------

def modified_blackbody_general_opacity_r(nu, z, theta, m=1, cmb=True):
    """
    General opacity modified blackbody model - continuum area
    
    :param nu: Rest frame frequency [Hz]
    :param z: Redshift
    :param theta: Fitting parameters
    :param m: Lensing magnification (Default = 1)
    :param cmb: Inclusion of CMB effects (Default = TRUE)
    :return: General opacity MBB model (continuum area)
    """
    # Uncouple parameters
    log_m, t, beta, r = theta

    # Include effects of the CMB
    if cmb:
        t_z = cmb_heating(z, t, beta)
        t = t_z

    # Retrieve dust mass, distance, blackbody and dust absorption coefficient
    mass = (10 ** log_m) * M_sun
    d = cosmo.luminosity_distance(z=z).to(u.m)/(1+z)
    bb = blackbody(nu, t)
    kappa = get_kappa(nu, beta)

    # Calculate the continuum area and solid angle
    r = (r * u.kpc).to(u.m)
    a = 4 * np.pi * (r ** 2)
    omega = a / (d ** 2)

    # Normalization from dust mass surface density and dust absorption coefficient
    sigma = mass / a
    tau = sigma * kappa
    modifier = (1 - np.exp(-tau))

    # Combine blackbody with normalization, modifiers and CMB
    mbb_rest = omega * m * modifier * bb
    if cmb:
        mbb_rest = mbb_rest * f_cmb(nu, t, z)

    mbb_obs = mbb_rest / (1 + z)
    return mbb_obs.value

# --------------------------------------------------

def modified_blackbody_general_opacity_r_powerlaw(nu, z, theta, m=1, cmb=True):
    """
    General opacity modified blackbody model with mid-IR power law - continuum area
    
    :param nu: Rest frame frequency [Hz]
    :param z: Redshift
    :param theta: Fitting parameters
    :param m: Lensing magnification (Default = 1)
    :param cmb: Inclusion of CMB effects (Default = TRUE)
    :return: General opacity MBB model with power law (continuum area)
    """
    # Uncouple parameters
    log_m, t, beta, alpha, r = theta
    theta_mbb = log_m, t, beta, r

    # Retrieve MBB model, find cutoff frequency, combine MBB with power law
    mbb = modified_blackbody_general_opacity_r(nu, z, theta_mbb, m, cmb)
    nu_c = find_cutoff_frequency(nu, mbb, alpha)
    mbb_nu_c = modified_blackbody_general_opacity_r(nu_c, z, theta_mbb, m, cmb)
    mbb_pl = add_powerlaw(nu, mbb, nu_c, mbb_nu_c, alpha)
    return mbb_pl

# --------------------------------------------------
# General Opacity Model (Optically thick wavelength)
# --------------------------------------------------

def modified_blackbody_general_opacity_lambda(nu, z, theta, m=1, cmb=True):
    """
    General opacity modified blackbody model - optically thick wavelength
    
    :param nu: Rest frame frequency [Hz]
    :param z: Redshift
    :param theta: Fitting parameters
    :param m: Lensing magnification (Default = 1)
    :param cmb: Inclusion of CMB effects (Default = TRUE)
    :return: General opacity MBB model (optically thick wavelength)
    """
    # Uncouple parameters
    log_m, t, beta, lambda_thick = theta

    # Include effects of the CMB
    if cmb:
        t_z = cmb_heating(z, t, beta)
        t = t_z

    # Retrieve dust mass, distance, blackbody and dust absorption coefficient
    mass = (10 ** log_m) * M_sun
    d = cosmo.luminosity_distance(z=z).to(u.m)/(1+z)
    bb = blackbody(nu, t)
    kappa = get_kappa(nu, beta)

    # Normalization from dust mass optical depth and dust absorption coefficient
    tau = get_tau(nu, lambda_thick, beta)
    a = (mass*kappa)/tau
    omega = a / (d ** 2)
    modifier = (1 - np.exp(-tau))

    # Combine blackbody with normalization, modifiers and CMB
    mbb_rest = omega * m * modifier * bb
    if cmb:
        mbb_rest = mbb_rest * f_cmb(nu, t, z)

    mbb_obs = mbb_rest / (1 + z)
    return mbb_obs.value

# --------------------------------------------------

def modified_blackbody_general_opacity_lambda_powerlaw(nu, z, theta, m=1, cmb=True):
    """
    General opacity modified blackbody model with mid-IR power law - optically thick wavelength
    
    :param nu: Rest frame frequency [Hz]
    :param z: Redshift
    :param theta: Fitting parameters
    :param m: Lensing magnification (Default = 1)
    :param cmb: Inclusion of CMB effects (Default = TRUE)
    :return: General opacity MBB model with power law (optically thick wavelength)
    """
    # Uncouple parameters
    log_m, t, beta, alpha, lambda_thick = theta
    theta_mbb = log_m, t, beta, lambda_thick

    # Retrieve MBB model, find cutoff frequency, combine MBB with power law
    mbb = modified_blackbody_general_opacity_lambda(nu, z, theta_mbb, m, cmb)
    nu_c = find_cutoff_frequency(nu, mbb, alpha)
    mbb_nu_c = modified_blackbody_general_opacity_lambda(nu_c, z, theta_mbb, m, cmb)
    mbb_pl = add_powerlaw(nu, mbb, nu_c, mbb_nu_c, alpha)
    return mbb_pl

# --------------------------------------------------
# Helper function for getting right model
# --------------------------------------------------

def get_model(nu, z, theta, m, opacity_model, powerlaw=True, cmb=True):
    """
    Returns the right MBB model depending on user inputs
    
    :param nu: Rest frame frequency [Hz]
    :param z: Redshift
    :param theta: Fitting parameters
    :param m: Lensing magnification
    :param opacity_model: Type of dust opacity model - "thin", "continuum_area" or "fixed_wave"
    :param powerlaw: Adds a power law function to the model (Default = TRUE)
    :param cmb: Inclusion of CMB effects (Default = TRUE)
    :return: MBB model as directed by user
    """
    if opacity_model == 'thin':
        if powerlaw:
            model = modified_blackbody_optically_thin_powerlaw(nu, z, theta, m, cmb)
        else:
            model = modified_blackbody_optically_thin(nu, z, theta, m, cmb)
    elif opacity_model == 'continuum_area':
        if powerlaw:
            model = modified_blackbody_general_opacity_r_powerlaw(nu, z, theta, m, cmb)
        else:
            model = modified_blackbody_general_opacity_r(nu, z, theta, m, cmb)
    elif opacity_model == 'fixed_wave':
        if powerlaw:
            model = modified_blackbody_general_opacity_lambda_powerlaw(nu, z, theta, m, cmb)
        else:
            model = modified_blackbody_general_opacity_lambda(nu, z, theta, m, cmb)
    return model