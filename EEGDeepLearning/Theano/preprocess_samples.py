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




def gen_images(locs, features, nGridPoints, normalize=True,
               augment=False, pca=False, stdMult=0.1, n_components=2, edgeless=False):
    """
    Generates EEG images given electrode locations in 2D space and multiple feature values for each electrode

    :param loc: An array with shape [n_electrodes, 2] containing X, Y
                        coordinates for each electrode.
    :param features: Feature matrix as [n_samples, n_features+1]
                                Features are as columns.
                                Features corresponding to each frequency band are concatenated.
                                (alpha1, alpha2, ..., beta1, beta2,...)
    :param nGridPoints: Number of pixels in the output images
    :param normalize:   Flag for whether to normalize each feature over all samples
    :param augment:     Flag for generating augmented images
    :param pca:         Flag for PCA based data augmentation
    :param stdMult:     Standard deviation of noise for augmentation
    :param n_components:Number of components in PCA to retain for augmentation
    :param edgeless:    If True generates edgeless images by adding artificial channels
                        at four corners of the image with value = 0 (default=False).
    :return:            Tensor of size [samples, colors, W, H] containing generated
                        images.
    """

    feat_array_temp = []
    nElectrodes = locs.shape[0]     # Number of electrodes

    # Test whether the feature vector length is divisible by number of electrodes (last column is the labels)
    assert features.shape[1] % nElectrodes == 0
    n_colors = features.shape[1] / nElectrodes
    for c in range(n_colors):
        feat_array_temp.append(features[:, c * nElectrodes : nElectrodes * (c+1)])


    if augment:
        if pca:
            for c in range(n_colors):
                feat_array_temp[c] = augment_EEG(feat_array_temp[c], stdMult, pca=True, n_components=n_components)
        else:
            for c in range(n_colors):
                feat_array_temp[c] = augment_EEG(feat_array_temp[c], stdMult, pca=False, n_components=n_components)

    nSamples = features.shape[0]
    # Interpolate the values
    grid_x, grid_y = np.mgrid[
                     min(locs[:, 0]):max(locs[:, 0]):nGridPoints*1j,
                     min(locs[:, 1]):max(locs[:, 1]):nGridPoints*1j
                     ]
    temp_interp = []
    for c in range(n_colors):
        temp_interp.append(np.zeros([nSamples, nGridPoints, nGridPoints]))

    # Generate edgeless images
    if edgeless:
        min_x, min_y = np.min(locs, axis=0)
        max_x, max_y = np.max(locs, axis=0)
        locs = np.append(locs, np.array([[min_x, min_y], [min_x, max_y],[max_x, min_y],[max_x, max_y]]),axis=0)
        for c in range(n_colors):
            feat_array_temp[c] = np.append(feat_array_temp[c], np.zeros((nSamples, 4)), axis=1)

    for i in xrange(nSamples):
        for c in range(n_colors):
            temp_interp[c][i, :, :] = griddata(locs, feat_array_temp[c][i, :], (grid_x, grid_y),
                                    method='cubic', fill_value=np.nan)
        print 'Interpolating {0}/{1}\r'.format(i+1, nSamples),

    for c in range(n_colors):
        if normalize:
            temp_interp[c][~np.isnan(temp_interp[c])] = \
                scale(temp_interp[c][~np.isnan(temp_interp[c])])
        temp_interp[c] = np.nan_to_num(temp_interp[c])

    return np.swapaxes(np.asarray(temp_interp), 0, 1)     # swap axes to have [samples, colors, W, H]





# # Plot 3D scatter plot of electrode locations
# fig = pl.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(caploc[:, 0], caploc[:, 1], caploc[:, 2], s=100)
# ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
# ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([]);
# ax.set_title('scatter plot of electrode locations')
# #pl.ion();
# #pl.show();

print "done"


mat = scipy.io.loadmat('Neuroscan_locs_orig')
caploc=mat['A']
new=[]
for element in np.arange(0,caploc.shape[0]):
    #print "cc"
    new.append(caploc[element][0])
    new.append(caploc[element][1])
locs=np.asanyarray(new).reshape(32,2)
print locs
print 'ok'


#
# #Plot Electrode locations
# fig = pl.figure()
# ax = fig.add_subplot(111)
# ax.scatter(locs[:, 0],locs[:, 1])
# ax.set_xlabel('X'); ax.set_ylabel('Y')
# ax.set_title('Polar Projection')
# #pl.ion();
# #pl.show()

# # Load power values
filename = 'FeatureMat_timeWinR'
#filename = 'C:\Users\Rakib\PycharmProjects\Theano\Sample data\FeatureMat_timeWin'
#mat= scipy.io.loadmat(filename)
#data=mat['features']
data, labels = load_data(filename)
#data = mat['featMat']
data=normalize(data, norm='l2', axis=1, copy=True)#axis  If 1, independently normalize each sample, otherwise (if 0) normalize each feature.
thetaFeats = data[:, :32]
alphaFeats = data[:, 32:64]
betaFeats = data[:, 64:96]
gammaFeats = data[:, 96:128]
#
if augment:
    if pca:
        thetaFeats = augment_EEG(thetaFeats, stdMult, pca=True, n_components=n_components)
        alphaFeats = augment_EEG(alphaFeats, stdMult, pca=True, n_components=n_components)
        betaFeats = augment_EEG(betaFeats, stdMult, pca=True, n_components=n_components)
        gammaFeats = augment_EEG(gammaFeats, stdMult, pca=True, n_components=n_components)
    else:
        thetaFeats = augment_EEG(thetaFeats, stdMult, pca=False, n_components=n_components)
        alphaFeats = augment_EEG(alphaFeats, stdMult, pca=False, n_components=n_components)
        gammaFeats = augment_EEG(gammaFeats, stdMult, pca=False, n_components=n_components)

#labels = data[:, -1]
nSamples = data.shape[0]

# Interpolate the values
grid_x, grid_y = np.mgrid[
                 min(locs[:, 0]):max(locs[:, 0]):nGridPoints*1j,
                 min(locs[:, 1]):max(locs[:, 1]):nGridPoints*1j
                 ]
thetaInterp = np.zeros([nSamples, nGridPoints, nGridPoints])
alphaInterp = np.zeros([nSamples, nGridPoints, nGridPoints])
betaInterp = np.zeros([nSamples, nGridPoints, nGridPoints])
gammaInterp = np.zeros([nSamples, nGridPoints, nGridPoints])

for i in xrange(nSamples):
    thetaInterp[i, :, :] = griddata(locs, thetaFeats[i, :], (grid_x, grid_y),
                                    method='cubic', fill_value=np.nan)
    alphaInterp[i, :, :] = griddata(locs, alphaFeats[i, :], (grid_x, grid_y),
                                    method='cubic', fill_value=np.nan)
    betaInterp[i, :, :] = griddata(locs, betaFeats[i, :], (grid_x, grid_y),
                                   method='cubic', fill_value=np.nan)
    gammaInterp[i, :, :] = griddata(locs, gammaFeats[i, :], (grid_x, grid_y),
                                   method='cubic', fill_value=np.nan)
    print 'Interpolating {0}/{1}\r'.format(i+1, nSamples),


# Byte scale to 0-255 range and substituting NaN with 0
# thetaInterp[~np.isnan(thetaInterp)] = bytescale(thetaInterp[~np.isnan(thetaInterp)])
thetaInterp[~np.isnan(thetaInterp)] = scale(thetaInterp[~np.isnan(thetaInterp)])
thetaInterp = np.nan_to_num(thetaInterp)
# alphaInterp[~np.isnan(alphaInterp)] = bytescale(alphaInterp[~np.isnan(alphaInterp)])
alphaInterp[~np.isnan(alphaInterp)] = scale(alphaInterp[~np.isnan(alphaInterp)])
alphaInterp = np.nan_to_num(alphaInterp)
# betaInterp[~np.isnan(betaInterp)] = bytescale(betaInterp[~np.isnan(betaInterp)])
betaInterp[~np.isnan(betaInterp)] = scale(betaInterp[~np.isnan(betaInterp)])
betaInterp = np.nan_to_num(betaInterp)
# gammaInterp[~np.isnan(betaInterp)] = bytescale(betaInterp[~np.isnan(betaInterp)])
gammaInterp[~np.isnan(gammaInterp)] = scale(gammaInterp[~np.isnan(gammaInterp)])
gammaInterp = np.nan_to_num(gammaInterp)

#
featureMatrix = np.zeros((nColors, nSamples, nGridPoints, nGridPoints))
featureMatrix[0, :, :, :] = thetaInterp
featureMatrix[1, :, :, :] = alphaInterp
featureMatrix[2, :, :, :] = betaInterp
featureMatrix[3, :, :, :] = gammaInterp
featureMatrix = np.swapaxes(featureMatrix, 0, 1)        # swap axes to have [samples, colors, W, H]
#
# # Save all data into mat file
scipy.io.savemat('EEG_images_32_flattened_locs', {'featMat': featureMatrix,
                                 'labels': labels})
# #
#
## ALL IMAGES SHOULD BE FLIPPED UPSIDE-DOWN ##
## USE np.flipud #############################
ind = 3
pl.figure()
# pl.subplot(321)
# negate the Y-Axis to make the image upside down
# pl.scatter(locs[:, 0], -locs[:,1], s=100, c=thetaFeats[ind, :].T); pl.title('Theta Electrodes')
pl.subplot(231)
pl.imshow(featureMatrix[ind, 0, :, :].T, cmap='Reds', vmin=np.min(featureMatrix[ind]), vmax=np.max(featureMatrix[ind])); pl.title('Theta')
pl.subplot(232)
pl.imshow(featureMatrix[ind, 1, :, :].T, cmap='Greens', vmin=np.min(featureMatrix[ind]), vmax=np.max(featureMatrix[ind])); pl.title('Alpha')
pl.subplot(233)
pl.imshow(featureMatrix[ind, 2, :, :].T, cmap='Blues', vmin=np.min(featureMatrix[ind]), vmax=np.max(featureMatrix[ind])); pl.title('Beta')
pl.subplot(234)
pl.imshow(featureMatrix[ind, 3, :, :].T, cmap='Blues', vmin=np.min(featureMatrix[ind]), vmax=np.max(featureMatrix[ind])); pl.title('gamma')
pl.subplot(235)
pl.imshow(np.swapaxes(np.rollaxis(featureMatrix[ind, :, :, :], 0, 3), 0, 1), vmin=np.min(featureMatrix[ind]), vmax=np.max(featureMatrix[ind])); pl.title('All')
pl.show()