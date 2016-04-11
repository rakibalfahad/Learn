"""
Reads the electrode locations and a matrix of EEG features. Generates
EEG images with three color channels.
Feature matrix is structured as [samples, features]. Features are
power in each frequency band over all electrodes
(theta-1, theta-2, ..., theta-64, alpha-1, ..., alpha-64, ...)
Locations file should contain coordinates for number of electrodes
existing in the features file.
"""
import numpy as np
import matplotlib.pyplot as pl #subplot(nrows, ncols, plot_number)
import scipy.io
from scipy.interpolate import griddata
from scipy.misc import bytescale
from sklearn.preprocessing import scale
from utilsP import augment_EEG,load_data
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import normalize
augment = False                 # Augment data
pca = False                     # Augment using PCA
stdMult = 0.1                   # Standard deviation of added noise
n_components = 2                # Number of components to keep
nGridPoints = 32                # Number of pixels in the image
nColors = 4                     # Number of color channels in the output image
nElectrodes = 32                # Number of electrodes

# Load electrode locations projected on a 2D surface
# mat = scipy.io.loadmat('Neuroscan_locs_polar_proj.mat')
# locs = mat['proj'] * [1, -1]    # reverse the Y axis to have the front on the top


