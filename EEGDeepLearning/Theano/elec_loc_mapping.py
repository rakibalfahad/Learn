
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
#from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as pl
import scipy.io
import math

data = scipy.io.loadmat('Neuroscan_locs_orig')

# Plot 3D scatter plot of electrode locations
fig = pl.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data['A'][:, 0], data['A'][:, 1], data['A'][:, 2], s=100)
# ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([]);
pl.ion();
pl.show();
# pl.savefig('electrodes_3d_locations', dpi=300)

# Plot 2D projection using perspective divide
# Z = data['A'][:, 2] + np.max(data['A'][:, 2])
# # Z = data['A'][:, 2] / np.max(data['A'][:, 2])
# fig = pl.figure()
# ax = fig.add_subplot(111)
# ax.scatter(data['A'][:, 0] / Z, data['A'][:, 1] / Z)
# ax.set_xlabel('X'); ax.set_ylabel('Y')
# pl.ion(); pl.show()

# Plot 2D polar projected points
data = scipy.io.loadmat('Z:\CVPIA\Pouya\SDrive\SkyDrive\Documents\Proposal\Programs\EEG_CNN\Neuroscan_locs_polar_proj.mat')

fig = pl.figure()
ax = fig.add_subplot(111)
ax.scatter(data['proj'][:, 0],-data['proj'][:, 1], s=100)
# ax.set_xlabel('X'); ax.set_ylabel('Y')
# ax.set_title('Azimuthal Projection')
ax.set_xticklabels([]); ax.set_yticklabels([]);
pl.ion();
# pl.savefig('electrodes_azim_projections', dpi=300)
pl.show()

#########################

#filename = 'EEG_images_32_timeWin'              # Name of the .mat file containing EEG images
#filename_aug = 'EEG_images_32_timeWin_aug_pca'  # Name of the .mat file containing augmented EEG images
#subjectsFilename = 'trials_subNums'




# Seems like the 2D polar projection from fieldtrip function (elproj) outputs the same projection as the Azimuth
# Equidist proj.
# Polar Azimuthal Equidistant Projection
data = scipy.io.loadmat('Z:\CVPIA\Pouya\SDrive\SkyDrive\Documents\Proposal\Programs\EEG_CNN\Neuroscan_locs_spherical.mat')
data = data['proj']

width = 10; lon_0 = -105; lat_0 = 40
m = Basemap(width=width,height=width,projection='aeqd',
            lat_0=90,lon_0=90)
proj = np.zeros(data.shape)
for i, elec in enumerate(data):
    proj[i, 0], proj[i, 1] = m(math.degrees(elec[0]), math.degrees(elec[1]))

fig = pl.figure()
ax = fig.add_subplot(111)
ax.scatter(proj[:, 0], proj[:, 1])
ax.set_xlabel('X'); ax.set_ylabel('Y')
ax.set_title('Azimuth Equidistant Projection')
pl.ioff();
pl.show()