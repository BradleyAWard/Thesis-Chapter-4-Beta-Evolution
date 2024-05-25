# --------------------------------------------------
# Imports
# --------------------------------------------------

import os
from pandas import read_csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --------------------------------------------------
# Load Data
# --------------------------------------------------

def full_loader(file_name, root=ROOT):
    """
    Loads a dataframe
    
    :param file_name: Name of the file
    :param root: Root directory
    :return: Datafile
    """

    file_path = root + '/data/' + str(file_name)
    data = read_csv(file_path)
    return data