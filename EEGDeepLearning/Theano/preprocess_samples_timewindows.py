import numpy as np
import math as m
import pandas as pd
import matplotlib.pyplot as pl
import scipy.io
from scipy.interpolate import griddata
from scipy.misc import bytescale
from sklearn.preprocessing import scale
from utilsP import augment_EEG
from utilsP import augment_EEG, cart2sph, pol2cart

def azim_proj(pos):
    """
    Computes the Azimuthal Equidistant Projection of input point in 3D Cartesian Coordinates.
    Imagine a plane being placed against (tangent to) a globe. If
    a light source inside the globe projects the graticule onto
    the plane the result would be a planar, or azimuthal, map
    projection.

    :param pos: position in 3D Cartesian coordinates
    :return: projected coordinates using Azimuthal Equidistant Projection
    """
    [r, elev, az] = cart2sph(pos[0], pos[1], pos[2])
    return pol2cart(az, m.pi / 2 - elev)

#A_10 = np.loadtxt(open("cap.csv","rb"),delimiter=",")
#d_10 = pd.read_csv("cap.csv") # load dataset
#a=d_10.drop('Electrode',1)
#carpos=a.iloc[1][:]
#A=([-83.20022999, -27.03339346, -3.054935226])
#carpos=[]
#for electrode in np.arange(1,a.shape[0]):
#        e=azim_proj(a.iloc[electrode][:])
#        carpos.append(e)

mat = scipy.io.loadmat('Neuroscan_locs_orig')
caploc=mat['A']

#caploc=pd.DataFrame(caploc)
#caploc=caploc.apply(azim_proj,axis=1)
new=[]
for element in np.arange(0,caploc.shape[0]):
    #print "cc"
    new.append(caploc[element][0])
    new.append(caploc[element][1])

locs=np.asanyarray(new).reshape(32,2)
print locs
print 'ok'









augment = False      # Augment data
pca = False
stdMult = 0.1
n_components = 2
nGridPoints = 32
nColors = 3                     # Number of color channels in the output image
numTimeWin = 7
#
# # Load electrode locations projected on a 2D surface
# mat = scipy.io.loadmat('Neuroscan_locs_flattened.mat')
# locs = mat['proj'] * [1, -1]    # reverse the Y axis to have the front on the top
#Plot Electrode locations
fig = pl.figure()
ax = fig.add_subplot(111)
ax.scatter(locs[:, 0],locs[:, 1])
ax.set_xlabel('X'); ax.set_ylabel('Y')
ax.set_title('Polar Projection')
#pl.ion();
pl.show()

# Load power values
dataMat = scipy.io.loadmat('FeatureMat_timeWinR')
# data = mat['features']
# labels = data[:, -1]
data = dataMat['featMat']
labels = dataMat['labels']
nSamples = data.shape[0]
#
timeFeatMat = []
for winNum in range(numTimeWin):
    print 'Processing window {0}/{1}'.format(winNum+1, numTimeWin)
    thetaFeats = data[:, winNum*192:winNum*192+64]
    alphaFeats = data[:, winNum*192+64:winNum*192+128]
    betaFeats = data[:, winNum*192+128:winNum*192+192]

    if augment:
        if pca:
            thetaFeats = augment_EEG(thetaFeats, stdMult, pca=True, n_components=n_components)
            alphaFeats = augment_EEG(alphaFeats, stdMult, pca=True, n_components=n_components)
            betaFeats = augment_EEG(betaFeats, stdMult, pca=True, n_components=n_components)
        else:
            thetaFeats = augment_EEG(thetaFeats, stdMult, pca=False, n_components=n_components)
            alphaFeats = augment_EEG(alphaFeats, stdMult, pca=False, n_components=n_components)
            betaFeats = augment_EEG(betaFeats, stdMult, pca=False, n_components=n_components)


    # Interpolate the values
    grid_x, grid_y = np.mgrid[
                     min(locs[:, 0]):max(locs[:, 0]):nGridPoints*1j,
                     min(locs[:, 1]):max(locs[:, 1]):nGridPoints*1j
                     ]
    thetaInterp = np.zeros([nSamples, nGridPoints, nGridPoints])
    alphaInterp = np.zeros([nSamples, nGridPoints, nGridPoints])
    betaInterp = np.zeros([nSamples, nGridPoints, nGridPoints])

    for i in xrange(nSamples):
        thetaInterp[i, :, :] = griddata(locs, thetaFeats[i, :], (grid_x, grid_y),
                                        method='cubic', fill_value=np.nan)
        alphaInterp[i, :, :] = griddata(locs, alphaFeats[i, :], (grid_x, grid_y),
                                        method='cubic', fill_value=np.nan)
        betaInterp[i, :, :] = griddata(locs, betaFeats[i, :], (grid_x, grid_y),
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

    featureMatrix = np.zeros((nColors, nSamples, nGridPoints, nGridPoints))
    featureMatrix[0, :, :, :] = thetaInterp
    featureMatrix[1, :, :, :] = alphaInterp
    featureMatrix[2, :, :, :] = betaInterp
    featureMatrix = np.swapaxes(featureMatrix, 0, 1)        # swap axes to have [samples, colors, W, H]
    timeFeatMat.append(featureMatrix)
# Save all data into mat file
scipy.io.savemat('EEG_images_32_timeWin_flattened_locs', {'featMat': np.asarray(timeFeatMat),
                                'labels': labels})

# # ind = 1600
# # pl.figure()
# # # pl.subplot(321)
# # # negate the Y-Axis to make the image upside down
# # # pl.scatter(locs[:, 0], -locs[:,1], s=100, c=thetaFeats[ind, :].T); pl.title('Theta Electrodes')
# # pl.subplot(221)
# # pl.imshow(featureMatrix[ind, 0, :, :].T); pl.title('Theta')
# # pl.subplot(222)
# # pl.imshow(featureMatrix[ind, 1, :, :].T); pl.title('Alpha')
# # pl.subplot(223)
# # pl.imshow(featureMatrix[ind, 2, :, :].T); pl.title('Beta')
# # pl.subplot(224)
# # pl.imshow(np.swapaxes(np.rollaxis(featureMatrix[ind, :, :, :], 0, 3), 0, 1)); pl.title('All')
# # pl.show()