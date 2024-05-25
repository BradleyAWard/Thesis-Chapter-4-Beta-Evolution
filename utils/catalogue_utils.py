# --------------------------------------------------
# Imports
# --------------------------------------------------

import utils
import dill
import numpy as np
import pandas as pd
from tqdm import tqdm

# --------------------------------------------------
# Create Catalogue and Catalogue Results
# --------------------------------------------------

def create_catalogue(data):
    """
    Creates a catalogue of sources from a data file
    
    :param data: Data file
    :return: Catalogue of sources
    """
    catalogue = []

    # For all objects in data, gather data and add source to a catalogue
    for obj in tqdm(range(len(data)), desc='Creating Sample'):

        # Gather ID and redshift
        identifier = data['Source'][obj]
        redshift = data['zspec'][obj]

        # Gather wavelengths, fluxes, flux errors and calibration percentages
        # 1 - SPT
        if data['spt_herbs'][obj] == 1:

            wavelengths_um_obs = utils.Data().wavelength_obs_um_spt
            calibrations = utils.Data().calibrations_spt
            fluxes = np.array([data['S_100'][obj], data['S_160'][obj], data['S_250'][obj],
                               data['S_350'][obj], data['S_500'][obj], data['S_870'][obj],
                               data['S_1400'][obj], data['S_2000'][obj], data['S_3000'][obj]]) / 1000
            flux_errors = np.array([data['E_100'][obj], data['E_160'][obj], data['E_250'][obj],
                                    data['E_350'][obj], data['E_500'][obj], data['E_870'][obj],
                                    data['E_1400'][obj], data['E_2000'][obj], data['E_3000'][obj]]) / 1000
        # 2 - HerBS
        elif data['spt_herbs'][obj] == 2:

            wavelengths_um_obs = utils.Data().wavelength_obs_um_herbs
            calibrations = utils.Data().calibrations_herbs
            fluxes = np.array([data['S_100'][obj], data['S_160'][obj], data['S_250'][obj],
                               data['S_350'][obj], data['S_500'][obj], data['S_850'][obj],
                               data['S_1200'][obj], data['S_2000'][obj], data['S_3000'][obj]]) / 1000
            flux_errors = np.array([data['E_100'][obj], data['E_160'][obj], data['E_250'][obj],
                                    data['E_350'][obj], data['E_500'][obj], data['E_850'][obj],
                                    data['E_1200'][obj], data['E_2000'][obj], data['E_3000'][obj]]) / 1000
        # Error handling
        else:
            raise ValueError('Does not have survey ID')

        # Gather magnification
        m = data['mag'][obj]

        # Add source to the catalogue
        source = utils.Source(identifier, wavelengths_um_obs, redshift, fluxes, flux_errors, m, calibrations)
        catalogue.append(source)
    return catalogue

# --------------------------------------------------

def catalogue_results(catalogue, fixparams=None, fixvalues=None, fixparams_array=None, fixvalues_array=None, opacity_models=None, opacity_models_array=None, powerlaw=True, cmb=True, min_obs_wavelength=None, min_rest_wavelength=None, max_obs_wavelength=None, remove_wavelength=None, verbose=True):
    """
    Fitting SEDs to a catalogue of sources
    
    :param catalogue: Catalogue of sources
    :param fixparams: Names of fixed parameters (Default = None)
    :param fixvalues: Values of fixed parameters (Default = None)
    :param fixparams_array: Array of names of fixed parameters (Default = None)
    :param fixvalues_array: Array of values of fixed parameters (Default = None)
    :param opacity_models: Opacity model for sources (Default = None)
    :param opacity_models_array: Array of opacity models for sources (Default = None)
    :param powerlaw: Add a power law to the SED model (Default = None)
    :param cmb: Include the effects of the CMB (Default = None)
    :param min_obs_wavelength: Set a minimum observed wavelength [microns] (Default = None)
    :param min_rest_wavelength: Set a minimum rest wavelength [microns] (Default = None)
    :param max_obs_wavelength: Set a maximum observed wavelength [microns] (Default = None)
    :param remove_wavelength: Remove photometry at a given wavelength [microns] (Default = None)
    :param verbose: Include all derived parameters (Default = True)
    :return: List of sources and dataframe of SED fitting results
    """
    n = len(catalogue)
    catalogue_summary = []

    for obj in tqdm(range(n), desc='SED Fitting for Catalogue'):

        # Check for fixed parameters
        if fixparams is not None:
            fixparam = [fixparams]
            fixvalue = [fixvalues]
        elif fixparams_array is not None:
            fixparam = [fixparams_array[obj]]
            fixvalue = [fixvalues_array[obj]]
        else:
            fixparam = []
            fixvalue = []

        # Check for opacity models, otherwise assume optically thin
        if opacity_models is not None:
            opacity_model = opacity_models
        elif opacity_models_array is not None:
            opacity_model = opacity_models_array[obj]
        else:
            opacity_model = 'thin'

        # Create a source, run SED fitting and append results
        source = catalogue[obj]
        source(opacity_model=opacity_model, powerlaw=powerlaw, cmb=cmb, fixparams=fixparam, fixvalues=fixvalue, min_obs_wavelength=min_obs_wavelength, min_rest_wavelength=min_rest_wavelength, max_obs_wavelength=max_obs_wavelength, remove_wavelength=remove_wavelength, progress=False, verbose=verbose)
        summary = source.results_summary()
        catalogue_summary.append(summary)
    catalogue_df = pd.DataFrame(catalogue_summary)
    return catalogue, catalogue_df


# --------------------------------------------------
# Save and Load Catalogue
# --------------------------------------------------

def save_catalogue(catalogue, folder_name, file_name, root=utils.ROOT):
    """
    Saves a catalogue
    
    :param catalogue: Catalogue to save
    :param folder_name: Name of directory
    :param file_name: Name of file
    :param root: Root location
    """
    with open(root + '/' + str(folder_name) + '/' + str(file_name), 'wb') as f:
        dill.dump(catalogue, f)

# --------------------------------------------------

def load_catalogue(folder_name, file_name, root=utils.ROOT):
    """
    Loads a catalogue
    
    :param folder_name: Name of directory
    :param file_name: Name of file
    :param root: Root location
    :return: Catalogue
    """
    with open(root + '/' + str(folder_name) + '/' + str(file_name), 'rb') as f:
        new_catalogue = dill.load(f)
    return new_catalogue
