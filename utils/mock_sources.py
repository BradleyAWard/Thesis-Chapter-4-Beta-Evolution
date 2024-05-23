# --------------------------------------------------
# Imports
# --------------------------------------------------

import utils
import numpy as np
import pandas as pd
from astropy.constants import c
from tqdm import tqdm
from scipy.optimize import root

# --------------------------------------------------
# Normalization of SED Conversion
# --------------------------------------------------

def get_logm(logm, loglir_fixed, t, beta, alpha, low_lim, high_lim, z, mag, cmb, luminosity_magnified=True):
    """
    Function to use in root algorithm for dust mass from an IR luminosity
    
    :param logm: Dust mass for dependent variable [log(X)]
    :param loglir_fixed: Target IR luminosity [log(Lsun)]
    :param t: Dust temperature [K]
    :param beta: Dust emissivity spectral index
    :param alpha: Mid-IR power law index
    :param low_lim: Low rest frame wavelength for IR luminosity [microns]
    :param high_lim: High rest frame wavelength for IR luminosity [microns]
    :param z: Redshift
    :param mag: Lensing magnification factor
    :param cmb: Include the effects of the CMB
    :param luminosity_magnified: Boolean choice for lensing magnification (Default = TRUE)
    :return: Difference between IR luminosity and target value
    """
    # Define an initial source
    initial_gal = utils.Source('initial', np.array([0]), z, np.array([0]), np.array([0]), mag, np.array([0]))
    initial_gal(fit_SED=False, opacity_model='thin', powerlaw=True, cmb=cmb)
    logmulir = initial_gal.ir_luminosity([logm, t, beta, alpha], lam_low_um_rest=low_lim, lam_high_um_rest=high_lim)

    # Apply lensing magnification if provided
    if luminosity_magnified:
        eq1 = logmulir-loglir_fixed
    else:
        loglir = np.log10((10**logmulir)/mag)
        eq1 = loglir-loglir_fixed
    return eq1

# --------------------------------------------------
# Simulations
# --------------------------------------------------

def simulation_setup(n, parameters, disable=False):
    """
    Sets up inputs for a simulations of sources
    
    :param n: Number of simulated galaxies
    :param parameters: Input parameters
    :param disable: Show progress of simulation (Default = FALSE)
    :return: Input parameters and simulated galaxies
    """
    # Setup arrays for input parameters and mock galaxies
    param_inputs_all = []
    fixed_param_inputs_all = []
    galaxies_all = []

    for _ in tqdm(range(n), desc='Creating Galaxies', disable=disable):
        np.random.seed()

        # Set up lists for input and fixed parameters
        param_names = []
        param_values = []
        fixed_names = []
        fixed_values = []

        # Check for input observed wavelengths in parameter file
        if 'wavelengths_obs_um' not in parameters:
            raise ValueError('Please add "wavelengths_obs_um" to parameters. (Wavelengths of observations in microns).')
        wavelengths_obs_um = parameters['wavelengths_obs_um']

        # Check for minimum detection thresholds in parameter file
        if 'minimum_detections' not in parameters:
            raise ValueError('Please add "minimum_detections" to parameters. (The minimum detection flux for each wavelength [Jy], this can be assigned to 0 to ignore detection limits).')
        minimum_detections = parameters['minimum_detections']

        # Check for calibration errors in parameter file
        if 'calibrations' not in parameters:
            raise ValueError('Please add "calibrations" to parameters. (The calibration error as a percentage for each wavelength, this can be assigned to 0 to ignore calibration errors).')
        calibrations = parameters['calibrations']

        # Check for CMB in parameter file
        if 'cmb' not in parameters:
            raise ValueError('Please add "cmb" to parameters. (True or False).')
        cmb = parameters['cmb']

        # Check for lensing magnification in parameter file
        if 'mag' not in parameters:
            raise ValueError('Please add "mag" to parameters. (Value of lensing magnification - leave as 1 if unlensed).')
        mag = parameters['mag']

        # Check for SNR in parameter file
        if 'snr' not in parameters:
            raise ValueError('Please add "snr" to parameters. (SNR mean and standard deviation for each wavelength).')
        snr = parameters['snr']

        # Obtain fixed redshift or select at random
        if 'fixed_z' in parameters:
            z_input = parameters['fixed_z']
        elif 'range_z' in parameters:
            z_lower, z_upper = parameters['range_z'][0], parameters['range_z'][1]
            z_input = np.random.uniform(z_lower, z_upper)
        else:
            raise ValueError("Please provide a range of redshifts 'range_z' or a fixed redshift 'fixed_z' in parameters." )

        # Obtain fixed dust temperature or select at random
        if 'fixed_t' in parameters:
            fixed_names.append('t')
            t_input = parameters['fixed_t']
            fixed_values.append(t_input)
        elif 'range_t' in parameters:
            t_lower, t_upper = parameters['range_t'][0], parameters['range_t'][1]
            t_input = np.random.uniform(t_lower, t_upper)
        else:
            raise ValueError("Please provide a range of dust temperatures 'range_t' or a fixed dust temperature 'fixed_t' in parameters." )

        # Obtain fixed dust emissivity spectral index or select at random
        if 'fixed_beta' in parameters:
            fixed_names.append('beta')
            beta_input = parameters['fixed_beta']
            fixed_values.append(beta_input)
        elif 'range_beta' in parameters:
            beta_lower, beta_upper = parameters['range_beta'][0], parameters['range_beta'][1]
            beta_input = np.random.uniform(beta_lower, beta_upper)
        else:
            raise ValueError("Please provide a range of dust emissivity indexes 'range_beta' or a fixed dust emissivity index 'fixed_beta' in parameters." )

        # Obtain fixed mid-IR power law index or select at random
        if 'fixed_alpha' in parameters:
            fixed_names.append('alpha')
            alpha_input = parameters['fixed_alpha']
            fixed_values.append(alpha_input)
        elif 'range_alpha' in parameters:
            alpha_lower, alpha_upper = parameters['range_alpha'][0], parameters['range_alpha'][1]
            alpha_input = np.random.uniform(alpha_lower, alpha_upper)
        else:
            raise ValueError("Please provide a range of mid-IR slopes 'range_alpha' or a fixed mid-IR slope 'fixed_alpha' in parameters." )

        # Obtain dust mass from range of LIR, fixed LIR, range of dust masses or fixed dust mass
        if 'range_logmulir' in parameters:
            if 'lir_limits' in parameters:
                lir_low, lir_high = parameters['lir_limits'][0], parameters['lir_limits'][1]
                logmulir_lower, logmulir_upper = parameters['range_logmulir'][0], parameters['range_logmulir'][1]
                logmulir_input = np.random.uniform(logmulir_lower, logmulir_upper)
                sol = root(get_logm, [10], args=(logmulir_input, t_input, beta_input, alpha_input, lir_low, lir_high, z_input, mag, cmb, True))
                logm_input = sol.x.item()
            else:
                raise ValueError('Please provide the limits of integration for IR luminosity "lir_limits" in parameters.')
        elif 'fixed_logmulir' in parameters:
            if 'lir_limits' in parameters:
                lir_low, lir_high = parameters['lir_limits'][0], parameters['lir_limits'][1]
                logmulir_input = parameters['fixed_logmulir']
                sol = root(get_logm, [10], args=(logmulir_input, t_input, beta_input, alpha_input, lir_low, lir_high, z_input, mag, cmb, True))
                fixed_names.append('log_m')
                logm_input = sol.x.item()
                fixed_values.append(logm_input)
            else:
                raise ValueError('Please provide the limits of integration for IR luminosity "lir_limits" in parameters.')
        elif 'range_loglir' in parameters:
            if 'lir_limits' in parameters:
                lir_low, lir_high = parameters['lir_limits'][0], parameters['lir_limits'][1]
                loglir_lower, loglir_upper = parameters['range_loglir'][0], parameters['range_loglir'][1]
                loglir_input = np.random.uniform(loglir_lower, loglir_upper)
                sol = root(get_logm, [10], args=(loglir_input, t_input, beta_input, alpha_input, lir_low, lir_high, z_input, mag, cmb, False))
                fixed_names.append('log_m')
                logm_input = sol.x.item()
                fixed_values.append(logm_input)
            else:
                raise ValueError('Please provide the limits of integration for IR luminosity "lir_limits" in parameters.')
        elif 'fixed_loglir' in parameters:
            if 'lir_limits' in parameters:
                lir_low, lir_high = parameters['lir_limits'][0], parameters['lir_limits'][1]
                loglir_input = parameters['fixed_loglir']
                sol = root(get_logm, [10], args=(loglir_input, t_input, beta_input, alpha_input, lir_low, lir_high, z_input, mag, cmb, False))
                logm_input = sol.x.item()
            else:
                raise ValueError('Please provide the limits of integration for IR luminosity "lir_limits" in parameters.')
        elif 'fixed_logm' in parameters:
            fixed_names.append('log_m')
            logm_input = parameters['fixed_logm']
            fixed_values.append(logm_input)
        elif 'range_logm' in parameters:
            logm_lower, logm_upper = parameters['range_logm'][0], parameters['range_logm'][1]
            logm_input = np.random.uniform(logm_lower, logm_upper)
        else:
            raise ValueError("Please provide a range of IR luminosities 'range_logmulir'/'range_loglir', a fixed IR luminosity 'fixed_logmulir'/'fixed_loglir', a range of dust masses 'range_logm' or a fixed dust mass 'fixed_logm' in parameters. (NOTE: a fixed IR luminosity only fixes the dust mass to its initial value, the final IR luminosity will change)." )

        # Add the input parameters to list
        param_names.append('logm')
        param_values.append(logm_input)
        param_names.append('t')
        param_values.append(t_input)
        param_names.append('beta')
        param_values.append(beta_input)
        param_names.append('alpha')
        param_values.append(alpha_input)

        # Create mock SED
        wavelengths_obs_m = wavelengths_obs_um*1e-6
        wavelengths_rest_m = wavelengths_obs_m/(1+z_input)
        frequencies_rest = c.value/wavelengths_rest_m
        mock_flux_jy = utils.get_model(frequencies_rest, z_input, param_values, mag, 'thin', powerlaw=True, cmb=cmb)

        # Perturb the SED
        snr_rand = np.array([np.abs(np.random.normal(snr[it][0], snr[it][1])) for it in range(len(snr))])
        mock_flux_error_jy = mock_flux_jy/snr_rand
        mock_flux_perturb_jy = np.abs(np.random.normal(mock_flux_jy, mock_flux_error_jy))

        # Apply detections
        # 0 - true SED not detected
        # 1 - perturbed SED not detected (but true SED is detected)
        # 2 - true and perturbed SED are both detected
        if any(mock_flux_jy < minimum_detections):
            detected = 0
        elif any(mock_flux_perturb_jy < minimum_detections):
            detected = 1
        else:
            detected = 2

        # Create galaxy from perturbed SED - derived parameters
        mock_gal = utils.Source('mock', wavelengths_obs_um, z_input, mock_flux_perturb_jy, mock_flux_error_jy, mag, calibrations)
        mock_gal(fit_SED=False, opacity_model='thin', powerlaw=True, cmb=cmb)
        galaxies_all.append(mock_gal)

        # Append derived input quantities
        logmulir_input = mock_gal.ir_luminosity(param_values, lam_low_um_rest=8, lam_high_um_rest=1000).value
        logmulfir_input = mock_gal.ir_luminosity(param_values, lam_low_um_rest=40, lam_high_um_rest=1000).value
        loglir_input = np.log10((10**logmulir_input)/mock_gal.magnification)
        loglfir_input = np.log10((10**logmulfir_input)/mock_gal.magnification)
        logmum_input = np.log10((10**logm_input)*mock_gal.magnification)
        peakwave_input = mock_gal.peak_wavelength(param_values)

        # Append derived parameters to input lists
        param_names.append('logmulir')
        param_values.append(logmulir_input)
        param_names.append('logmulfir')
        param_values.append(logmulfir_input)
        param_names.append('loglir')
        param_values.append(loglir_input)
        param_names.append('loglfir')
        param_values.append(loglfir_input)
        param_names.append('logmum')
        param_values.append(logmum_input)
        param_names.append('peakwave')
        param_values.append(peakwave_input)
        param_names.append('detected')
        param_values.append(detected)

        # Add input parameters to list for all simulated galaxies
        param_inputs_all.append(param_values)
        fixed_param_inputs_all.append(fixed_values)

    # Create dataframes from the fixed and variable input parameters for all simulated galaxies
    inputs = pd.DataFrame(param_inputs_all, columns=param_names)
    fixed_inputs = pd.DataFrame(fixed_param_inputs_all, columns=fixed_names)
    return inputs, fixed_inputs, galaxies_all

# --------------------------------------------------

def simulation(n, parameters, name, nwalkers=25, niters=2000, sigma=1e-4, burn_in=500, disable=False):
    """
    Runs SED fitting simulation for a set of mock galaxies
    
    :param n: Number of simulated galaxies
    :param parameters: Parameter file for simulations
    :param name: Name of simulation when saved
    :param nwalkers: Number of walkers in MC (Default = 25)
    :param niters: Number of MC iterations (Default = 2000)
    :param sigma: Exploration parameter of MC (Default = 1e-4)
    :param burn_in: Number of initial steps in MC that are ignored (Default = 500)
    :param disable: Show progress of simulation (Default = FALSE)
    :return: Inputs, outputs and mock galaxies
    """
    # Set up lists for output parameters and output galaxies
    outputs, galaxies_output = [], []
    inputs, fixed_inputs, galaxies_input = simulation_setup(n, parameters)

    # Run simulation for each galaxy created
    for gal in tqdm(range(len(galaxies_input)), desc='Simulation', disable=disable):
        # Create galaxy, fit SED and gather results
        galaxy = galaxies_input[gal]
        galaxy('thin', fixparams=fixed_inputs.columns.tolist(), fixvalues=fixed_inputs.iloc[gal].tolist(), cmb=parameters['cmb'], powerlaw=True, nwalkers=nwalkers, niters=niters, sigma=sigma, burn_in=burn_in, progress=False, verbose=False)
        galaxy_output = galaxy.results_summary()

        # Gather derived quantities
        galaxy_output['logmulir'] = galaxy.ir_luminosity(galaxy.best_theta, lam_low_um_rest=8, lam_high_um_rest=1000).value
        galaxy_output['logmulfir'] = galaxy.ir_luminosity(galaxy.best_theta, lam_low_um_rest=40, lam_high_um_rest=1000).value
        galaxy_output['loglir'] = np.log10((10**galaxy_output['logmulir'])/galaxy.magnification)
        galaxy_output['loglfir'] = np.log10((10**galaxy_output['logmulfir'])/galaxy.magnification)
        galaxy_output['logmum'] = np.log10((10**galaxy_output['logm'])*galaxy.magnification)
        galaxy_output['peakwave'] = galaxy.peak_wavelength(galaxy.best_theta)

        # Append outputs to lists
        galaxies_output.append(galaxy)
        outputs.append(galaxy_output)
    outputs = pd.DataFrame(outputs)

    # Save simulations results
    utils.save_catalogue(inputs, 'simulation_results', name+'_inputs')
    utils.save_catalogue(outputs, 'simulation_results', name+'_outputs')
    utils.save_catalogue(galaxies_output, 'simulation_results', name+'_mock_galaxies')

    return inputs, outputs, galaxies_output