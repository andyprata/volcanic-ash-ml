# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 14:36:05 2021

@author: Mikey Waugh

A script which calls functions from Functions1.py to fit and visualise ML
models for classifying volcanic ash from Nishinoshima using data from the
Himawari-8 satellite.

*Note to self*
-Need to check data and polygons do actually match.
-Scene locations:
    top_row, bottom_row = 1290, 1590
    left_column, right_column = 2520, 2920
    for
    (1st) datetime(2020, 8, 1, 5)
    (2nd) datetime(2020, 8, 1, 12)
    (3rd) datetime(2020, 7, 30, 16) - Not sure the data and polygon match here.

    top_row, bottom_row = 1100, 1400
    left_column, right_column = 2520, 2920
    for
    (4th) datetime(2020, 7, 28, 5)
-For the whole strip of Himawari data:
    top_row, bottom_row = 1100, 1650
    left_column, right_column = 600, 5200
"""

import matplotlib
import netCDF4

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from Functions1 import *
from datetime import datetime
from sklearn import svm, metrics, neighbors, tree
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC, LinearSVC
from skopt import BayesSearchCV
import numpy as np
import sys
# import winsound
import cv2 as cv
import pandas as pd


# This (and the line at the bottom of the document) find the code runtime
start = datetime.now()

create_VAG = False
plot_BTDs = False
plot_ash = True
want_coefficient_plot = True
predicted_plot_flag = True
denoised_image_flag = True

band_waves_dict = {'B01': '455 nm', 'B02': '510 nm', 'B03': '645 nm',
                   'B04': '860 nm', 'B05': '1610 nm', 'B06': '2260 nm',
                   'B07': '3.85 $\mu$m', 'B08': '6.25 $\mu$m',
                   'B09': '6.95 $\mu$m', 'B10': '7.35 $\mu$m',
                   'B11': '8.60 $\mu$m', 'B12': '9.63 $\mu$m',
                   'B13': '10.45 $\mu$m', 'B14': '11.20 $\mu$m',
                   'B15': '12.35 $\mu$m', 'B16': '13.30 $\mu$m'}

#############################################################################
############################# CREATING DATASETS #############################
#############################################################################

# Focus only on subset
# r1, r2 = 1300, 1600
# c1, c2 = 2500, 2900

# Try full disk
r1, r2 = 0, 5500
c1, c2 = 0, 5500

# Read in VAAC Polygon Data
path_data = '/home/aprata/PycharmProjects/volcanic-ash-ml/data/'
# Case 1 (Daytime - Original scene)
polygon_filename = "20200801_28409600_0184_Text.html"
sat_datetime = datetime(2020, 8, 1, 5)

# # Case 2 (Night-time  - 7 hours after original scene)
# polygon_filename = "20200801_28409600_0185_Text.html"
# sat_datetime = datetime(2020, 8, 1, 12)

process_second_time = False
if process_second_time:
    sat_datetime2 = datetime(2020, 7, 28, 5)
    #sat_datetime2 = datetime(2020, 8, 1, 12)



polygon_fileloc = path_data + "vaa/"
sat_location = path_data + 'himawari8/full_disk/'
lats_polygon, lons_polygon = poly_latlon(polygon_filename, polygon_fileloc, create_VAG)

# Read in Himawari data
filelist = glob.glob(sat_location + "*" + sat_datetime.strftime("%Y%m%d_%H%M") + "*.DAT")
scn = Scene(filelist, reader='ahi_hsd')

# Load all channels
scn.load(scn.available_dataset_names())

# Resample to 2 km resolution
lcn = scn.resample(scn.coarsest_area(), resampler='native')

if process_second_time:
    filelist2 = glob.glob(sat_location + "*" + sat_datetime2.strftime("%Y%m%d_%H%M") + "*.DAT")
    scn2 = Scene(filelist2, reader='ahi_hsd')
    scn2.load(scn2.available_dataset_names())
    lcn2 = scn2.resample(scn2.coarsest_area(), resampler='native')

# Basis for ash detection
BTD = lcn['B14'].data[r1:r2, c1:c2] - lcn['B15'].data[r1:r2, c1:c2]

# Water vapour correction (note: increasing b reduces the water vapour correction)
T11 = lcn['B14'].data[r1:r2, c1:c2]
b = 4.5
wv = np.exp(6. * (T11 / 320.) - b)
BTD_wv = BTD - wv

# Extract pixels inside VAAC polygon
polygon = [(lon, lat) for lon, lat in zip(lons_polygon, lats_polygon)]

# Get lons and lats for each point in our area? Not sure if these are correct
# or if I need to convert them? They seem to work...?
lons_fd, lats_fd = lcn['B14'].attrs['area'].get_lonlats()
lons, lats = lons_fd[r1:r2, c1:c2], lats_fd[r1:r2, c1:c2]

# Create the array of BTDs for just ash
points = np.vstack((lons.flatten(), lats.flatten())).T
p = Path(polygon)
grid = p.contains_points(points)
Mask = grid.reshape(lats.shape)
BTD_wv_ma = np.ma.array(BTD_wv, mask=Mask == False)

# Masks BTDs greater than or equal to zero
BTD_wv_ma = np.ma.masked_greater_equal(BTD_wv_ma, 0)

# Format data into scikit-learn vector format
# Decide on which channels (i.e. 'features') to use
features = np.copy(scn.available_dataset_names())  # ALL chans
#features = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06']  # VIS-only (not including 3.9 micron)
# features = ['B07', 'B08', 'B09', 'B10', 'B11', 'B12',
#             'B13', 'B14', 'B15', 'B16']  # IR-only (including 3.9 micron)

# Initialise X with first band
X = np.reshape(lcn[features[0]].values[r1:r2, c1:c2], (-1, 1))
if process_second_time:
    X2 = np.reshape(lcn2[features[0]].values[r1:r2, c1:c2], (-1, 1))

# Add all other bands
for band in features:
    # skip first band
    if band != features[0]:
        X = np.hstack((X, np.reshape(lcn[band].values[r1:r2, c1:c2], (-1, 1))))
        if process_second_time:
            X2 = np.hstack((X2, np.reshape(lcn2[band].values[r1:r2, c1:c2], (-1, 1))))

# add BTD as a feature
# X = np.hstack((X, np.reshape(lcn['B14'].values[r1:r2, c1:c2]-lcn['B15'].values[r1:r2, c1:c2], (-1, 1))))

# Now create y by replacing masked BTD values with zero then setting all
# values less than zero to 1 (ash) and all more than zero to 0 (not ash)
# before reshaping
y = BTD_wv_ma.filled(0)
y = np.where(y < 0, 1, 0).ravel()

# Drop any row of data that contains a NaN/Inf
# Convert to dataframe
df_X = pd.DataFrame(data=X)
if process_second_time:
    df_X2 = pd.DataFrame(data=X2)
df_y = pd.DataFrame(data=y)

qq = (np.isfinite(df_X)).all(axis=1)

if process_second_time:
    qq2 = (np.isfinite(df_X2)).all(axis=1)

# Convert back to numpy array
X_filtered = df_X[qq].to_numpy()
if process_second_time:
    X2_filtered = df_X2[qq2].to_numpy()
y_filtered = df_y[qq].to_numpy()

# n = 1
# y_pcum_2d = np.zeros((T11.shape))

# for i in range(n):
#     print(i)
# Get 8000 randomly sampled 'ash' pixels
#ash_ind = np.random.choice(np.squeeze(np.where(y_filtered[:, 0] == 1)), 8000)

# Use all ash data points
ash_ind = np.squeeze(np.where(y_filtered[:, 0] == 1))

# Fraction of pixels identified as ash
ash_frac = len(ash_ind)/y_filtered.shape[0]

# Maximum number of 'not ash' pixels is
not_ash_max = y_filtered.shape[0] - len(ash_ind)

# Get 10x randomly sampled 'not ash' pixels
not_ash_factor = 100

# Total number of ash pixels
not_ash_num = not_ash_factor*len(ash_ind)

if not_ash_num >= not_ash_max:
    raise ValueError('Not ash factor set too high.')

not_ash_ind = np.random.choice(np.squeeze(np.where(y_filtered[:, 0] == 0)), not_ash_num)
reduced_ind = np.concatenate((ash_ind, not_ash_ind))

X_reduced = np.array(X_filtered[reduced_ind, :])
y_reduced = np.array(y_filtered[reduced_ind, :])

# We shuffle and split the reduced dataset into the training and test data.
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y_reduced, test_size=0.2)

# We now train our model
C_val = 1
svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C=C_val, dual=False))
])
svm_clf.fit(X_train, y_train)

# Classify the scene
yp = svm_clf.predict(X_filtered)

# Classify a completely different scene (same eruption)
if process_second_time:
    yp2 = svm_clf.predict(X2_filtered)

# Plot the prediction
y_pred = np.zeros_like(y)
y_pred[qq] = yp
y_pred_2d = y_pred.reshape(r2-r1, c2-c1)
plt.imshow(y_pred_2d)
plt.title('C='+str(C_val))
plt.show()

# plt.imshow(y_prob_2d[r1:r2, c1:c2])
# plt.imshow(y_prob_2d)

plt.figure()
importances = svm_clf['linear_svc'].coef_
features_names = []
for f in features:
    features_names.append(f + ' (' + band_waves_dict[f] + ')')
importances, features_names = zip(*sorted(zip(importances[0], features_names)))
plt.barh(range(len(features_names)), importances)
plt.yticks(range(len(features_names)), features_names)
plt.title('Linear SVC Coefficients')
plt.tight_layout()
plt.show()

# plot random pixels used to classify 'not ash'
plt.figure()
y_na_filtered = np.ones_like(y_filtered) * -1
y_na_filtered[ash_ind] = 1
y_na_filtered[not_ash_ind] = 0

y_na = np.ones_like(y) * -2
y_na[qq] = y_na_filtered[:, 0]
plt.imshow(y_na.reshape(r2-r1, c2-c1))
plt.tight_layout()
plt.show()

# Plot the prediction of scene 2
if process_second_time:
    plt.figure()
    y2_pred = np.zeros_like(y)
    y2_pred[qq2] = yp2
    y2_pred_2d = y2_pred.reshape(r2-r1, c2-c1)
    plt.imshow(y2_pred_2d)
    plt.title('C='+str(C_val) + ', New scene prediction')
    plt.show()
