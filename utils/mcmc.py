# --------------------------------------------------
# Imports
# --------------------------------------------------

import utils
import numpy as np
import emcee

# --------------------------------------------------
# Monte Carlo Markov Chain SED Fitting
# --------------------------------------------------

def mcmc(frequencies_rest, redshift, fluxes, flux_errors, m, opacity_model, fixed_params, fixed_values, gaussian_prior, gaussian_prior_values, powerlaw, cmb, nwalkers, niters, sigma, progress):
    """
    Monte Carlo Markov Chain SED fitting function
    
    :param frequencies_rest: Rest frame frequencies [Hz]
    :param redshift: Redshift of galaxy
    :param fluxes: Flux densities of galaxy [Jy]
    :param flux_errors: Flux density errors of galaxy [Jy]
    :param m: Gravitational lensing magnification
    :param opacity_model: Type of dust opacity model - "thin", "continuum_area" or "fixed_wave"
    :param fixed_params: List of fixed parameter names
    :param fixed_values: List of fixed parameter values
    :param gaussian_prior: List of parameter names for Gaussian prior
    :param gaussian_prior_values: List of parameter values for Gaussian prior
    :param powerlaw: Adds a power law to model
    :param cmb: Inclusion of CMB effects
    :param nwalkers: Number of walkers in MC 
    :param niters: Number of MC iterations
    :param sigma: Exploration parameter of MC
    :param progress: Show progress
    :return: Variable parameter dictionary and MCMC sampler
    """
    # Error handling
    if len(fixed_params) != len(fixed_values):
        raise ValueError('Requires an equal number of fixed parameter names to values.')

    # Define the lowest possible temperature (the CMB temperature at redshift z)
    t_cmb = 2.725 * (1 + redshift)

    # Define the lower and upper bounds of each parameter plus the initial value of the MCMC chain
    log_m_lower, log_m_initial, log_m_upper = 5, 9, 13
    t_lower, t_initial, t_upper = t_cmb, 50., 100.
    beta_lower, beta_initial, beta_upper = 0.5, 2, 6.
    alpha_lower, alpha_initial, alpha_upper = 0.5, 2., 8.
    r_lower, r_initial, r_upper = 0, 1, 6
    lambda_thick_lower, lambda_thick_initial, lambda_thick_upper = 40, 100, 250

    # Setting up the arrays for the initial, lower and upper values of parameters
    all_parameters = ["log_m", "t", "beta", "alpha", "r", "lambda_thick"]
    all_parameters_initial = [log_m_initial, t_initial, beta_initial, alpha_initial, r_initial, lambda_thick_initial]
    all_parameters_lower = [log_m_lower, t_lower, beta_lower, alpha_lower, r_lower, lambda_thick_lower]
    all_parameters_upper = [log_m_upper, t_upper, beta_upper, alpha_upper, r_upper, lambda_thick_upper]

    # Select parameters depending on the model
    opacity_models = ['thin', 'continuum_area', 'fixed_wave']
    if opacity_model not in opacity_models:
        raise ValueError('Invalid opacity model. Please select one of "thin", "continuum area" (for a fixed value give "r" in [kpc]) or "fixed_wave" (for a fixed value give "lambda_thick" in [microns].')

    if opacity_model == 'thin':
        if powerlaw:
            parameters_full = ["log_m", "t", "beta", "alpha"]
        else:
            parameters_full = ["log_m", "t", "beta"]
    elif opacity_model == 'continuum_area':
        if powerlaw:
            parameters_full = ["log_m", "t", "beta", "alpha", "r"]
        else:
            parameters_full = ["log_m", "t", "beta", "r"]
    elif opacity_model == 'fixed_wave':
        if powerlaw:
            parameters_full = ["log_m", "t", "beta", "alpha", "lambda_thick"]
        else:
            parameters_full = ["log_m", "t", "beta", "lambda_thick"]

    # Define initial, lower and upper sets of parameters
    parameters_initial = np.array(
        [param for param, name in zip(all_parameters_initial, all_parameters) if name in parameters_full])
    parameters_lower = np.array(
        [param for param, name in zip(all_parameters_lower, all_parameters) if name in parameters_full])
    parameters_upper = np.array(
        [param for param, name in zip(all_parameters_upper, all_parameters) if name in parameters_full])

    # Further error handling
    for it in fixed_params:
        if it not in parameters_full:
            raise ValueError(str(it) + ' not a valid parameter name. Must be one of: ' + str(parameters_full))
    if len(fixed_params) >= len(parameters_full):
        raise ValueError('At least one parameter must be allowed to vary')

    # Identify which parameters the user has defined as fixed and variable
    index_fix = [parameters_full.index(par) for par in fixed_params]
    varyparams = [param for param in parameters_full if param not in fixed_params]
    index_vary = [parameters_full.index(par) for par in varyparams]

    # Initialize the variable parameters
    initial_vary = parameters_initial[index_vary]
    lower_vary = parameters_lower[index_vary]
    upper_vary = parameters_upper[index_vary]
    ndim = len(initial_vary)
    pos = initial_vary + (sigma * np.random.randn(nwalkers, ndim))
    parameters_initial[index_fix] = fixed_values
    data = (parameters_initial, frequencies_rest, redshift, fluxes, flux_errors)

    # Function returns the model values at each step given current parameters
    def model(theta, theta_full, nu, z):
        if opacity_model == 'thin':
            if powerlaw:
                 [log_m, t, beta, alpha] = theta_full
            else:
                [log_m, t, beta] = theta_full
        elif opacity_model == 'continuum_area':
            if powerlaw:
                [log_m, t, beta, alpha, r] = theta_full
            else:
                [log_m, t, beta] = theta_full
        elif opacity_model == 'fixed_wave':
            if powerlaw:
                [log_m, t, beta, alpha, lambda_thick] = theta_full
            else:
                [log_m, t, beta, lambda_thick] = theta_full
        y_model = utils.get_model(nu, z, theta_full, m, opacity_model, powerlaw, cmb)
        theta_full[index_vary] = theta
        return y_model

    # Function returns the log likelihood of the current state of the model in the chain
    def log_likelihood(theta, theta_full, nu, z, flux, flux_error):
        ymodel = model(theta, theta_full, nu, z)
        log_like = -0.5 * np.sum(((flux - ymodel) / flux_error) ** 2)
        return log_like

    # Returns the log likelihood of the prior
    def log_prior(theta, theta_full):
        if opacity_model == 'thin':
            if powerlaw:
                 [log_m, t, beta, alpha] = theta_full
            else:
                [log_m, t, beta] = theta_full
        elif opacity_model == 'continuum_area':
            if powerlaw:
                [log_m, t, beta, alpha, r] = theta_full
            else:
                [log_m, t, beta] = theta_full
        elif opacity_model == 'fixed_wave':
            if powerlaw:
                [log_m, t, beta, alpha, lambda_thick] = theta_full
            else:
                [log_m, t, beta, lambda_thick] = theta_full
        theta_full[index_vary] = theta

        # Apply a Gaussian prior
        if gaussian_prior is not None:
            idx_gaussian = [varyparams.index(param) for param in gaussian_prior]
            idx_flat = [varyparams.index(param) for param in varyparams if param not in gaussian_prior]
            theta_gaussian = [theta[idx] for idx in idx_gaussian]
            theta_flat = [theta[idx] for idx in idx_flat]
        
        # Assume flat priors
        else:
            theta_flat = theta
        for it in range(len(theta_flat)):
            if not lower_vary[it] < theta_flat[it] < upper_vary[it]:
                return -np.inf
        prior = 0
        for it in range(len(theta_gaussian)):
            mu = gaussian_prior_values[it][0]
            sigma = gaussian_prior_values[it][1]
            prior += np.log(1.0/(np.sqrt(2*np.pi)*sigma))-0.5*(theta_gaussian[it]-mu)**2/sigma**2
        return prior

    # Returns the log probability, including the prior
    def log_probability(theta, theta_full, nu, z, flux, flux_error):
        lp = log_prior(theta, theta_full)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta, theta_full, nu, z, flux, flux_error)

    # Run the MCMC
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=data, parameter_names=None)
    sampler.run_mcmc(pos, niters, progress=progress)
    return sampler, varyparams
