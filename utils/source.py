# --------------------------------------------------
# Imports
# --------------------------------------------------

import utils
import numpy as np
import corner
import astropy.units as u
from astropy.constants import c, M_sun, L_sun
from matplotlib.patches import Ellipse
from astropy.cosmology import Planck18 as cosmo

# --------------------------------------------------
# Data Class
# --------------------------------------------------

class Data:
    """
    A class to define observed wavelengths and calibration errors
    """
    def __init__(self):
        self.wavelength_obs_um_spt = np.array([100, 160, 250, 350, 500, 870, 1400, 2000, 3000])
        self.wavelength_obs_um_herbs = np.array([100, 160, 250, 350, 500, 850, 1200, 2000, 3000])
        self.calibrations_spt = np.array([7, 7, 5.5, 5.5, 5.5, 12, 7, 7, 10])
        self.calibrations_herbs = np.array([7, 7, 5.5, 5.5, 5.5, 5, 10, 10, 10])

# --------------------------------------------------
# Source Class
# --------------------------------------------------

class Source:

    def __init__(self, name: str, wavelength_obs_um: list, redshift: float, flux_jy: list, fluxerr_jy: list, magnification: float, calibration_percent: list):
        self.name = name
        self.redshift = redshift
        self.wavelength_obs_um = wavelength_obs_um
        self.flux_jy = flux_jy
        self.fluxerr_jy = fluxerr_jy
        self.magnification = magnification
        self.calibration_percent = calibration_percent

        # Remove data points with no photometry
        indices = np.logical_not(np.logical_or(np.isnan(self.flux_jy), np.isnan(self.fluxerr_jy)))
        self.wavelength_obs_um, self.flux_jy, self.fluxerr_jy, self.calibration_percent = self.wavelength_obs_um[indices], self.flux_jy[indices], self.fluxerr_jy[indices], self.calibration_percent[indices]

        # Add calibration errors
        calibration_err = [(percent / 100) * flux for percent, flux in zip(self.calibration_percent, self.flux_jy)]
        self.fluxerr_jy = np.array([np.sqrt((flux_err ** 2) + (cal_err ** 2)) for flux_err, cal_err in zip(self.fluxerr_jy, calibration_err)])

        # Conversions
        self.wavelength_obs_m = self.wavelength_obs_um * 1e-6
        self.wavelength_rest_m = self.wavelength_obs_m / (1 + self.redshift)
        self.wavelength_rest_um = self.wavelength_rest_m / 1e-6
        self.frequencies_obs = c.value / self.wavelength_obs_m
        self.frequencies_rest = c.value / self.wavelength_rest_m

        # Save full photometry for cases when not all used
        self.wavelength_obs_um_full, self.wavelength_rest_um_full, self.flux_jy_full, self.fluxerr_jy_full = self.wavelength_obs_um, self.wavelength_rest_um, self.flux_jy, self.fluxerr_jy
        self.frequencies_obs_full, self.frequencies_rest_full = self.frequencies_obs, self.frequencies_rest

        # Find the minimum wavelength we have photometry for
        self.min_wavelength_obs = min(self.wavelength_obs_um)


    def __call__(self, opacity_model=None, min_rest_wavelength=None, min_obs_wavelength=None, max_obs_wavelength=None, remove_wavelength=None, fixparams=[], fixvalues=[], gaussian_prior=[], gaussian_prior_values=[], powerlaw=True, cmb=True, fit_SED=True, nwalkers=25, niters=2000, sigma=1e-4, burn_in=500, progress=True, verbose=True):

        # Run MCMC and obtain samples
        self.opacity_model = opacity_model
        self.fixparams = fixparams
        self.fixvalues = fixvalues
        self.gaussian_prior = gaussian_prior
        self.gaussian_prior_values = gaussian_prior_values
        self.powerlaw = powerlaw
        self.cmb = cmb
        self.burn_in = burn_in
        self.verbose = verbose

        # Check of a valid opacity model
        opacity_models = ['thin', 'continuum_area', 'fixed_wave']
        if self.opacity_model not in opacity_models:
            raise ValueError('Invalid opacity model. Please select one of "thin", "continuum_area" (for a fixed value give "r" in [kpc]) or "fixed_wave" (for a fixed value give "lambda_thick" in [microns]).')

        # Remove photometry if minimum rest wavelength set
        if min_rest_wavelength:
            wavelength_rest_um = (self.wavelength_obs_um / (1 + self.redshift))
            wavelength_rest_um_used = np.array([wave for wave, flux, flux_error in zip(wavelength_rest_um, self.flux_jy, self.fluxerr_jy) if wave > min_rest_wavelength])
            flux_jy = np.array([flux for wave, flux, flux_error in zip(wavelength_rest_um, self.flux_jy, self.fluxerr_jy) if wave > min_rest_wavelength])
            flux_err_jy = np.array([flux_error for wave, flux, flux_error in zip(wavelength_rest_um, self.flux_jy, self.fluxerr_jy) if wave > min_rest_wavelength])
            wavelength_obs_um_used = wavelength_rest_um_used * (1 + self.redshift)
            self.wavelength_obs_um, self.flux_jy, self.fluxerr_jy = wavelength_obs_um_used, flux_jy, flux_err_jy

        # Remove photometry if minimum observed wavelength set
        elif min_obs_wavelength:
            wavelength_obs_um_used = np.array([wave for wave, flux, flux_error in zip(self.wavelength_obs_um, self.flux_jy, self.fluxerr_jy) if wave >= min_obs_wavelength])
            flux_jy = np.array([flux for wave, flux, flux_error in zip(self.wavelength_obs_um, self.flux_jy, self.fluxerr_jy) if wave >= min_obs_wavelength])
            flux_err_jy = np.array([flux_error for wave, flux, flux_error in zip(self.wavelength_obs_um, self.flux_jy, self.fluxerr_jy) if wave >= min_obs_wavelength])
            self.wavelength_obs_um, self.flux_jy, self.fluxerr_jy = wavelength_obs_um_used, flux_jy, flux_err_jy

        # Remove photometry if maximum observed wavelength set
        elif max_obs_wavelength:
            wavelength_obs_um_used = np.array([wave for wave, flux, flux_error in zip(self.wavelength_obs_um, self.flux_jy, self.fluxerr_jy) if wave <= max_obs_wavelength])
            flux_jy = np.array([flux for wave, flux, flux_error in zip(self.wavelength_obs_um, self.flux_jy, self.fluxerr_jy) if wave <= max_obs_wavelength])
            flux_err_jy = np.array([flux_error for wave, flux, flux_error in zip(self.wavelength_obs_um, self.flux_jy, self.fluxerr_jy) if wave <= max_obs_wavelength])
            self.wavelength_obs_um, self.flux_jy, self.fluxerr_jy = wavelength_obs_um_used, flux_jy, flux_err_jy

        # Remove given photometry
        elif remove_wavelength:
            remove_wavelength = np.array(remove_wavelength)
            wavelength_obs_um_used = np.array([wave for wave, flux, flux_error in zip(self.wavelength_obs_um, self.flux_jy, self.fluxerr_jy) if wave not in remove_wavelength])
            flux_jy = np.array([flux for wave, flux, flux_error in zip(self.wavelength_obs_um, self.flux_jy, self.fluxerr_jy) if wave not in remove_wavelength])
            flux_err_jy = np.array([flux_error for wave, flux, flux_error in zip(self.wavelength_obs_um, self.flux_jy, self.fluxerr_jy) if wave not in remove_wavelength])
            self.wavelength_obs_um, self.flux_jy, self.fluxerr_jy = wavelength_obs_um_used, flux_jy, flux_err_jy

        # Conversions
        self.wavelength_obs_m = self.wavelength_obs_um * 1e-6
        self.wavelength_rest_um = self.wavelength_obs_um / (1 + self.redshift)
        self.wavelength_rest_m = self.wavelength_obs_m / (1 + self.redshift)
        self.frequencies_obs = c.value / self.wavelength_obs_m
        self.frequencies_rest = c.value / self.wavelength_rest_m
        self.luminosity_distance = cosmo.luminosity_distance(z=self.redshift).to(u.m)

        if fit_SED:
            self.sampler, self.varyparams = utils.mcmc(self.frequencies_rest, self.redshift, self.flux_jy, self.fluxerr_jy, self.magnification, self.opacity_model, self.fixparams, self.fixvalues, self.gaussian_prior, self.gaussian_prior_values, self.powerlaw, self.cmb, nwalkers=nwalkers, niters=niters, sigma=sigma, progress=progress)

    # Function returns the SED
    def sed(self, frequencies_rest, theta):
        y_model = utils.get_model(frequencies_rest, self.redshift, theta, m=self.magnification, opacity_model=self.opacity_model, powerlaw=self.powerlaw, cmb=self.cmb)
        return y_model

    # Returns the IR luminosity for a given set of theta
    def ir_luminosity(self, theta, lam_low_um_rest=8, lam_high_um_rest=1000):

        lam_low_um_rest, lam_high_um_rest = lam_low_um_rest*u.micron, lam_high_um_rest*u.micron
        wave_range_rest_um = np.linspace(1, 5000, 100000) * u.micron
        wave_range_rest_m = wave_range_rest_um.to(u.m)
        freq_range_rest = c / wave_range_rest_m
        idx = np.where((wave_range_rest_um >= lam_low_um_rest) & (wave_range_rest_um <= lam_high_um_rest))

        diff_freq = np.diff(freq_range_rest)
        diff_freq = np.append(diff_freq, diff_freq[-1])

        sed_obs_integral = self.sed(freq_range_rest[idx].value, theta) * u.Jy
        sed_rest_integral = sed_obs_integral/(1+self.redshift)
        integral = np.sum(-sed_rest_integral*diff_freq[idx])

        d_L = cosmo.luminosity_distance(z=self.redshift).to(u.m)
        l_watt = (4 * np.pi * (d_L ** 2) * integral).to(u.Watt)
        l_sun = l_watt / L_sun
        log_l_sun = np.log10(l_sun)
        return log_l_sun

    # Returns the peak wavelength for a given set of theta
    def peak_wavelength(self, theta):
        wave_range_obs_um = np.linspace(5, 10000, 5000)
        wave_range_obs_m = wave_range_obs_um*1e-6
        wave_range_rest_um = wave_range_obs_um/(1+self.redshift)
        wave_range_rest_m = wave_range_obs_m/(1+self.redshift)
        frequency_range_rest = c.value/wave_range_rest_m
        sed = self.sed(frequency_range_rest, theta)
        peak_index = np.argmax(sed)
        return wave_range_rest_um[peak_index]

    # Obtain sampler and add fixed and derived values
    def get_full_sampler(self):

        if self.opacity_model == "thin":
            if self.powerlaw:
                params = ["log_m", "t", "beta", "alpha"]
            else:
                params = ["log_m", "t", "beta"]
        if self.opacity_model == "continuum_area":
            if self.powerlaw:
                params = ["log_m", "t", "beta", "alpha", "r"]
            else:
                params = ["log_m", "t", "beta", "r"]
        elif self.opacity_model == "fixed_wave":
            if self.powerlaw:
                params = ["log_m", "t", "beta", "alpha", "lambda_thick"]
            else:
                params = ["log_m", "t", "beta", "lambda_thick"]
        self.params = params
        self.params_full = self.params.copy()

        flat_samples = self.sampler.get_chain(discard=self.burn_in, flat=True)
        n = len(flat_samples)
        sample_theta = np.zeros((n, len(self.params)))
        vary_param_idx = [self.params.index(self.varyparams[param_idx]) for param_idx in range(len(self.varyparams))]
        fix_param_idx = [self.params.index(self.fixparams[param_idx]) for param_idx in range(len(self.fixparams))]
        sample_theta[:, vary_param_idx] = flat_samples
        sample_theta[:, fix_param_idx] = self.fixvalues
        sample_theta_vary = sample_theta[:, vary_param_idx]
        params_vary = [self.params[idx] for idx in vary_param_idx]

        sample_full = sample_theta
        sample_full_vary = sample_theta_vary
        self.params_theta_vary = params_vary
        self.params_full_vary = self.params_theta_vary.copy()

        # Radius or peak wavelength, depending on which model was used
        if self.opacity_model == 'continuum_area':
            lambda_thick = []
            for it in range(len(sample_theta)):
                log_m_val = sample_full[it, self.params_full.index('log_m')]
                beta_val = sample_full[it, self.params_full.index('beta')]
                r_val = sample_full[it, self.params_full.index('r')]
                m_kg = ((10 ** log_m_val) * M_sun).value
                r_m = (r_val * u.kpc).to(u.m).value
                lam_thick_val = utils.LAM_UM_REF * (((utils.KAPPA_REF * m_kg) / (4 * np.pi * (r_m ** 2))) ** (1 / beta_val))
                lambda_thick.append(lam_thick_val)
            sample_full = np.hstack((sample_full, np.zeros((n, 1))))
            sample_full_vary = np.hstack((sample_full_vary, np.zeros((n, 1))))
            sample_full[:, -1] = lambda_thick
            sample_full_vary[:, -1] = lambda_thick
            self.params_full.append("lambda_thick")
            self.params_full_vary.append("lambda_thick")

        elif self.opacity_model == 'fixed_wave':
            r = []
            for it in range(len(sample_theta)):
                log_m_val = sample_full[it, self.params_full.index('log_m')]
                beta_val = sample_full[it, self.params_full.index('beta')]
                lambda_thick_val = sample_full[it, self.params_full.index('lambda_thick')]
                m_kg = ((10 ** log_m_val) * M_sun).value
                r_val = ((np.sqrt(((utils.KAPPA_REF*m_kg)/(4*np.pi))*(1/((lambda_thick_val/utils.LAM_UM_REF)**beta_val)))*u.m).to(u.kpc)).value
                r.append(r_val)
            sample_full = np.hstack((sample_full, np.zeros((n, 1))))
            sample_full_vary = np.hstack((sample_full_vary, np.zeros((n, 1))))
            sample_full[:, -1] = r
            sample_full_vary[:, -1] = r
            self.params_full.append("r")
            self.params_full_vary.append("r")

        if self.verbose:

            # Add log(mu M)
            log_mu_m = []
            for it in range(len(sample_theta)):
                log_m_val = sample_full[it, self.params_full.index('log_m')]
                log_mu_m_val = np.log10((10**log_m_val)*self.magnification)
                log_mu_m.append(log_mu_m_val)
            sample_full = np.hstack((sample_full, np.zeros((n, 1))))
            sample_full_vary = np.hstack((sample_full_vary, np.zeros((n, 1))))
            sample_full[:, -1] = log_mu_m
            sample_full_vary[:, -1] = log_mu_m
            self.params_full.append("log_mu_m")
            self.params_full_vary.append("log_mu_m")

            # Add log(LIR) and log(mu LIR)
            log_mu_lir = []
            log_mu_lfir = []
            log_lir = []
            log_lfir = []
            for it in range(len(sample_theta)):
                log_mu_lir_val = self.ir_luminosity(sample_theta[it], lam_low_um_rest=8, lam_high_um_rest=1000)
                log_mu_lfir_val = self.ir_luminosity(sample_theta[it], lam_low_um_rest=40, lam_high_um_rest=1000)
                log_lir_val = np.log10((10**log_mu_lir_val)/self.magnification)
                log_lfir_val = np.log10((10 ** log_mu_lfir_val) / self.magnification)
                log_mu_lir.append(log_mu_lir_val)
                log_mu_lfir.append(log_mu_lfir_val)
                log_lir.append(log_lir_val)
                log_lfir.append(log_lfir_val)
            sample_full = np.hstack((sample_full, np.zeros((n, 1))))
            sample_full_vary = np.hstack((sample_full_vary, np.zeros((n, 1))))
            sample_full[:, -1] = log_mu_lir
            sample_full_vary[:, -1] = log_mu_lir
            self.params_full.append("log_mu_LIR")
            self.params_full_vary.append("log_mu_LIR")
            sample_full = np.hstack((sample_full, np.zeros((n, 1))))
            sample_full_vary = np.hstack((sample_full_vary, np.zeros((n, 1))))
            sample_full[:, -1] = log_mu_lfir
            sample_full_vary[:, -1] = log_mu_lfir
            self.params_full.append("log_mu_LFIR")
            self.params_full_vary.append("log_mu_LFIR")
            sample_full = np.hstack((sample_full, np.zeros((n, 1))))
            sample_full_vary = np.hstack((sample_full_vary, np.zeros((n, 1))))
            sample_full[:, -1] = log_lir
            sample_full_vary[:, -1] = log_lir
            self.params_full.append("log_LIR")
            self.params_full_vary.append("log_LIR")
            sample_full = np.hstack((sample_full, np.zeros((n, 1))))
            sample_full_vary = np.hstack((sample_full_vary, np.zeros((n, 1))))
            sample_full[:, -1] = log_lfir
            sample_full_vary[:, -1] = log_lfir
            self.params_full.append("log_LFIR")
            self.params_full_vary.append("log_LFIR")

            # Add peak wavelength
            peak_wave = []
            for it in range(len(sample_theta)):
                peak_wave_val = self.peak_wavelength(sample_theta[it])
                peak_wave.append(peak_wave_val)
            sample_full = np.hstack((sample_full, np.zeros((n, 1))))
            sample_full_vary = np.hstack((sample_full_vary, np.zeros((n, 1))))
            sample_full[:, -1] = peak_wave
            sample_full_vary[:, -1] = peak_wave
            self.params_full.append("peak_wave")
            self.params_full_vary.append("peak_wave")

        # Define the sampler for theta and all parameters
        self.sample_theta = sample_theta
        self.sample_full = sample_full

        # Define the sampler for theta and all parameters, providing they are variable
        self.sample_theta_vary = sample_theta_vary
        self.sample_full_vary = sample_full_vary

        return self.sample_theta, self.sample_full

    # Function returns the best fitting parameters
    def get_parameters(self, low_percentile=16, best_percentile=50, high_percentile=84):

        # For each variable parameter calculate the median and the lower and upper percentiles
        best_params = {}
        best_theta = []
        sample_theta, sample_total = self.get_full_sampler()
        ndims_full = len(self.params_full)
        ndims = len(self.params)
        for i in range(ndims_full):
            mcmc_param = np.percentile(sample_total[:, i], [low_percentile, best_percentile, high_percentile])
            mcmc_param_errors = np.diff(mcmc_param)
            error_low, median, error_high = mcmc_param_errors[0], mcmc_param[1], mcmc_param_errors[1]
            best_params[self.params_full[i]] = (error_low, median, error_high)
            if i < ndims:
                best_theta.append(median)

        self.best_params = best_params
        self.best_theta = best_theta

        return best_params, best_theta

    # Function returns n samples of the posterior
    def sample_walkers(self, n):

        # Select random draws from the sampler
        sample_theta_idx = np.random.choice(len(self.sample_theta), n)
        n_sample_theta = [self.sample_theta[i] for i in sample_theta_idx]

        sample_total_idx = np.random.choice(len(self.sample_full), n)
        n_sample_full = [self.sample_full[i] for i in sample_total_idx]

        return n_sample_theta, n_sample_full

    # Returns the residuals for a given model
    def residuals(self, theta=None):
        if theta is None:
            theta = self.best_theta
        y_model = self.sed(self.frequencies_rest, theta)
        residual = (self.flux_jy-y_model)/self.flux_jy
        return residual

    # Returns the chi-squared statistic for a given model
    def chi_squared(self, theta=None):
        if theta is None:
            theta = self.best_theta
        y_model = self.sed(self.frequencies_rest, theta)
        chi2 = np.sum([((i - j) ** 2) / (k ** 2) for i, j, k, in zip(y_model, self.flux_jy, self.fluxerr_jy)])
        return chi2

    # Returns the reduced chi-squared statistic for a given model
    def reduced_chi_squared(self, theta=None):
        if theta is None:
            theta = self.best_theta
        chi_squared = self.chi_squared(theta)
        reduced_chi_squared = chi_squared / (len(self.frequencies_rest) - len(self.varyparams))
        return reduced_chi_squared

    # Returns the corner plot of posterior distributions for the variable parameters
    def corner_plot(self):
        if self.verbose:
            fig = corner.corner(self.sample_full_vary, labels=self.params_full_vary, bins=30, plot_contour=True, color='b')
        else:
            fig = corner.corner(self.sample_theta_vary, labels=self.params_theta_vary, bins=30, plot_contour=True, color='b')

    # Returns the nth sigma confidence ellipse between beta and dust temperature
    def beta_temperature_contours(self, n_std=3):

        # Sample the posterior distributions of beta and dust temperature
        t_idx = self.params_full_vary.index("t")
        beta_idx = self.params_full_vary.index("beta")
        t_sample = self.sample_full_vary[:, t_idx]
        beta_sample = self.sample_full_vary[:, beta_idx]

        # Calculate the covariance of the two parameters and the eigenvalues
        cov = np.cov(t_sample, beta_sample)
        lambda_, v = np.linalg.eig(cov)
        lambda_ = np.sqrt(lambda_)

        # Generate the confidence ellipse using the width, height and angle
        ell = Ellipse(xy=(np.mean(t_sample), np.mean(beta_sample)), width=lambda_[0] * n_std * 2, height=lambda_[1] * n_std * 2, angle=np.degrees(np.arctan2(*v[:, 0][::-1])))
        ell.set_facecolor('none')
        ell.set_edgecolor('k')
        return ell

    # Returns the nth sigma confidence ellipse between beta and peak wavelength
    def beta_peak_wavelength_contours(self, n_std=3):

        # Sample the posterior distributions of beta and peak_wavelength
        peak_wave_idx = self.params_full_vary.index("peak_wave")
        beta_idx = self.params_full_vary.index("beta")
        peak_wave_sample = self.sample_full_vary[:, peak_wave_idx]
        beta_sample = self.sample_full_vary[:, beta_idx]

        # Calculate the covariance of the two parameters and the eigenvalues
        cov = np.cov(peak_wave_sample, beta_sample)
        lambda_, v = np.linalg.eig(cov)
        lambda_ = np.sqrt(lambda_)

        # Generate the confidence ellipse using the width, height and angle
        ell = Ellipse(xy=(np.mean(peak_wave_sample), np.mean(beta_sample)), width=lambda_[0] * n_std * 2, height=lambda_[1] * n_std * 2, angle=np.degrees(np.arctan2(*v[:, 0][::-1])))
        ell.set_facecolor('none')
        ell.set_edgecolor('k')
        return ell

    # Returns a summary of the most important values
    def results_summary(self):
        best_params, best_theta = self.get_parameters()
        t_low, t, t_high = best_params['t']
        log_m_low, log_m, log_m_high = best_params['log_m']
        beta_low, beta, beta_high = best_params['beta']
        if self.opacity_model == 'thin':
            summary = {'id': self.name,
                       'z': self.redshift,
                       'logm_low': log_m_low, 'logm': log_m, 'logm_high': log_m_high,
                       't_low': t_low, 't': t, 't_high': t_high,
                       'beta_low': beta_low, 'beta': beta, 'beta_high': beta_high,
                       'chi': self.chi_squared(),
                       'chi_red': self.reduced_chi_squared()}

        elif self.opacity_model == 'fixed_wave' or self.opacity_model == 'continuum_area':
            lambda_thick_low, lambda_thick, lambda_thick_high = best_params['lambda_thick']
            r_low, r, r_high = best_params['r']
            summary = {'id': self.name,
                       'z': self.redshift,
                       'logm_low': log_m_low, 'logm': log_m, 'logm_high': log_m_high,
                       't_low': t_low, 't': t, 't_high': t_high,
                       'beta_low': beta_low, 'beta': beta, 'beta_high': beta_high,
                       'lambda_thick_low': lambda_thick_low, 'lambda_thick': lambda_thick, 'lambda_thick_high': lambda_thick_high,
                       'r_low': r_low, 'r': r, 'r_high': r_high,
                       'chi': self.chi_squared(),
                       'chi_red': self.reduced_chi_squared()}

        if self.verbose:
            log_mu_lir_low, log_mu_lir, log_mu_lir_high = best_params['log_mu_LIR']
            log_mu_lfir_low, log_mu_lfir, log_mu_lfir_high = best_params['log_mu_LFIR']
            log_lir_low, log_lir, log_lir_high = best_params['log_LIR']
            log_lfir_low, log_lfir, log_lfir_high = best_params['log_LFIR']
            log_mu_m_low, log_mu_m, log_mu_m_high = best_params['log_mu_m']
            peak_wave_low, peak_wave, peak_wave_high = best_params['peak_wave']

            summary.setdefault("logmulir_low", log_mu_lir_low)
            summary.setdefault("logmulir", log_mu_lir)
            summary.setdefault("logmulir_high", log_mu_lir_high)

            summary.setdefault("logmulfir_low", log_mu_lfir_low)
            summary.setdefault("logmulfir", log_mu_lfir)
            summary.setdefault("logmulfir_high", log_mu_lfir_high)

            summary.setdefault("loglir_low", log_lir_low)
            summary.setdefault("loglir", log_lir)
            summary.setdefault("loglir_high", log_lir_high)

            summary.setdefault("loglfir_low", log_lfir_low)
            summary.setdefault("loglfir", log_lfir)
            summary.setdefault("loglfir_high", log_lfir_high)

            summary.setdefault("logmum_low", log_mu_m_low)
            summary.setdefault("logmum", log_mu_m)
            summary.setdefault("logmum_high", log_mu_m_high)

            summary.setdefault("peakwave_low", peak_wave_low)
            summary.setdefault("peakwave", peak_wave)
            summary.setdefault("peakwave_high", peak_wave_high)

        return summary
