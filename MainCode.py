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

from Functions1 import *
from datetime import datetime
from sklearn import svm, metrics, neighbors, tree
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
import numpy as np
import matplotlib.pyplot as plt
import sys
#import winsound
import cv2 as cv

# This (and the line at the bottom of the document) find the code runtime
start = datetime.now()

create_VAG = False
plot_BTDs = False
plot_ash = True
want_coefficient_plot = True
predicted_plot_flag = True
denoised_image_flag = True

#############################################################################
############################# CREATING DATASETS #############################
#############################################################################

# Define the variables we need to call our function
polygon_filename = "20200801_28409600_0184_Text.html"
polygon_fileloc = "./data/vaa/"
sat_datetime = datetime(2020, 8, 1, 5)
#sat_location = '/data/himawari8/20200801_0500/'
sat_location = './data/himawari8/'
top_row, bottom_row = 1100, 1650
left_column, right_column = 600, 5200

# Call the function to create X and y then print the outputs.
X_full, y_full, ash_num = dataset(polygon_filename, polygon_fileloc,
                                  sat_datetime, sat_location, top_row,
                                  bottom_row, left_column, right_column,
                                  create_VAG=create_VAG, plot_BTDs=plot_BTDs,
                                  plot_ash=plot_ash)

print("Number of data points that contain ash:", ash_num)

# We can also load in a separate scene. This could be used to test the model
# or to expand the training data.
second_scene = True
if second_scene:
    polygon_filename2 = '20200801_28409600_0185_Text.html'
    polygon_fileloc2 = './data/vaa/'
    sat_datetime2 = datetime(2020, 8, 1, 12)
    sat_location2 = './data/himawari8/'
    top_row2, bottom_row2 = 1100, 1650
    left_column2, right_column2 = 600, 5200

    X_full2, y_full2, ash_num2 = dataset(polygon_filename2, polygon_fileloc2,
                                         sat_datetime2, sat_location2,
                                         top_row2, bottom_row2, left_column2,
                                         right_column2, create_VAG=create_VAG,
                                         plot_BTDs=plot_BTDs,
                                         plot_ash=plot_ash)
    
    print("Number of data points that contain ash in second scene:", ash_num2)

# We load in a third scene here.
third_scene = True
if third_scene:
    polygon_filename3 = '20200730_28409600_0178_Text.html'
    polygon_fileloc3 = './data/vaa/'
    sat_datetime3 = datetime(2020, 7, 30, 16)
    sat_location3 = './data/himawari8/'
    top_row3, bottom_row3 = 1100, 1650
    left_column3, right_column3 = 600, 5200

    X_full3, y_full3, ash_num3 = dataset(polygon_filename3, polygon_fileloc3,
                                         sat_datetime3, sat_location3,
                                         top_row3, bottom_row3, left_column3,
                                         right_column3, create_VAG=create_VAG,
                                         plot_BTDs=plot_BTDs,
                                         plot_ash=plot_ash)
    
    print("Number of data points that contain ash in third scene:", ash_num3)

# We load in a fourth scene here.
fourth_scene = True
if fourth_scene:
    polygon_filename4 = '20200728_28409600_0168_Text.html'
    polygon_fileloc4 = './data/vaa/'
    sat_datetime4 = datetime(2020, 7, 28, 5)
    sat_location4 = './data/himawari8/'
    top_row4, bottom_row4 = 1100, 1650
    left_column4, right_column4 = 600, 5200

    X_full4, y_full4, ash_num4 = dataset(polygon_filename4, polygon_fileloc4,
                                         sat_datetime4, sat_location4,
                                         top_row4, bottom_row4, left_column4,
                                         right_column4, create_VAG=create_VAG,
                                         plot_BTDs=plot_BTDs,
                                         plot_ash=plot_ash)
    
    print("Number of data points that contain ash in fourth scene:", ash_num4)


# Turn any stray nan results into zeros so they have no contribution to SVC.
# It is probably better to instead change an entire row to something (possibly
# zero) but that is more effort and there don't seem to be many nan values so
# this is probably fine for now to prevent errors.
X_full = np.nan_to_num(X_full)
if second_scene:
    X_full2 = np.nan_to_num(X_full2)
if third_scene:
    X_full3 = np.nan_to_num(X_full3)
if fourth_scene:
    X_full4 = np.nan_to_num(X_full4)


# This is te ratio of not ash to ash points included in the test/train data
# for each frame
ratio=20
# We can now reduce the first dataset
X, y = reduce_dataset(X_full, y_full, ash_num, ratio=ratio)

# Here we can choose to add the 'stray' bit which keeps getting classified as
# ash to our training and test data to see if it improves the model. We need
# to be careful though and make sure we don't overfit our model to the
# original scene. Also this only works for the scene at the datetime given.
include_weird_region = True
if include_weird_region and sat_datetime == datetime(2020, 8, 1, 5):
    X, y = add_weird_region(X_full, y_full, X, y)


# Here we can join the datasets for our scenes.
add_scene_2 = True
if second_scene and add_scene_2:
    X2, y2 = reduce_dataset(X_full2, y_full2, ash_num2, ratio=ratio)
    X, y = np.concatenate((X, X2)), np.concatenate((y, y2))


# Here we can join the datasets for our scenes.
add_scene_3 = True
if third_scene and add_scene_3:
    X3, y3 = reduce_dataset(X_full3, y_full3, ash_num3, ratio=ratio)
    X, y = np.concatenate((X, X3)), np.concatenate((y, y3))


# Here we can join the datasets for our scenes.
add_scene_4 = False
if fourth_scene and add_scene_4:
    X4, y4 = reduce_dataset(X_full4, y_full4, ash_num4, ratio=ratio)
    X, y = np.concatenate((X, X4)), np.concatenate((y, y4))
    


#############################################################################
############################ SCIKIT-LEARN MODELS ############################
#############################################################################
# Below I have some scikit-learn models.

# Here is the first one. We use a train/test split and scale the data to a
# Gaussian around zero to create a linear Support Vector Classifier.
print("\nFirst Attempt (Train/Test Split (Linear/Non-Linear) SVC):")
# Set if we want to fit a model with a linear or non-linear kernel.
Linear=True
# We shuffle and split the reduced dataset into the training and test data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# We now create our pipeline. We transform our data using the standard scaler
# before applying an SVC. (Not sure if we need the pipeline within a
# pipeline?)
num_transformer = Pipeline(steps=[('scaler', StandardScaler())])
if Linear:
    pipe = Pipeline(steps=[('preprocessor', num_transformer),
                            ('classifier', svm.LinearSVC(C=10000, dual=False))])
else:
    pipe = Pipeline(steps=[('preprocessor', num_transformer),
                            ('classifier', svm.SVC(C=10000.0, gamma=0.16,
                                                  degree=4, kernel='rbf',
                                                  random_state=1))])

# pipe = Pipeline(steps=[('preprocessor', num_transformer),
#                         ('classifier', neighbors.KNeighborsClassifier())])

# pipe = Pipeline(steps=[('preprocessor', num_transformer),
#                         ('classifier', tree.DecisionTreeClassifier())])

# We train our model
pipe.fit(X_train, y_train)

# Here we plot a graph of the coefficients for our linear SVC. A positive
# coefficient means the a more positive band value contributes more to 'not
# ash' while a negative coefficient means a more positive band value
# contributes more to 'ash'.
if Linear and want_coefficient_plot:
    # This if statement obviously doesn't work if we don't have all the 
    # desired channels but I usually do so it's fine for a quick fix
    if 'desired_channels' in locals():
        features_names = list(desired_channels)
        coefficients_plot(pipe, features_names)
    else: 
        coefficients_plot(pipe)


# We test our model and produce a classification report, confusion matrix and
# model score on our test data
print("\nHere we try our model for the test data:")

predicted = pipe.predict(X_test)

print(f"Classification report for classifier on test data:\n"
      f"{metrics.classification_report(y_test, predicted)}\n")

print("Model score on test data:", pipe.score(X_test, y_test))

disp1 = metrics.plot_confusion_matrix(pipe, X_test, y_test)
disp1.figure_.suptitle("Confusion Matrix on Test Data")
plt.show()


# Now we can test our model on the entire original scene.
print("\nHere we try our model for the whole original scene:")

predicted = pipe.predict(X_full)

print(f"Classification report for classifier on original scene:\n"
      f"{metrics.classification_report(y_full, predicted)}\n")

print("Model score on whole original scene:", pipe.score(X_full, y_full))

disp1 = metrics.plot_confusion_matrix(pipe, X_full, y_full)
disp1.figure_.suptitle("Confusion Matrix on Original Scene")
plt.show()

# Here we plot the predictions of our linear kernel SVC against the 'true'
# ash locations. We additionally plot the additional points predicted to see
# where our model overestimates ash, as well as the points missed to see where
# our model underestimates ash.
im_width, im_height = right_column-left_column, bottom_row-top_row
if predicted_plot_flag:
    predicted_plots(im_width, im_height, pipe, X_full, y_full)
print("How about some extra colour here of the training data? Also how about",
      "adding in averages to the Himawari data?")


# Let's now test the model on the second scene.
# Here we plot a confusion matrix for the whole scene.
if second_scene:
    print("\nHere we try our model for the whole second scene:")
    
    predicted2 = pipe.predict(X_full2)
    
    print(f"Classification report for classifier on second scene:\n"
          f"{metrics.classification_report(y_full2, predicted2)}\n")
    
    print("Model score on second scene:", pipe.score(X_full2, y_full2))
    
    disp1 = metrics.plot_confusion_matrix(pipe, X_full2, y_full2)
    disp1.figure_.suptitle("Confusion Matrix for Second Scene")
    plt.show()
    
    
    # Here we can plot a graph of the predicted vs. true ash points for this
    # second scene.
    im_width2, im_height2 = right_column2-left_column2, bottom_row2-top_row2
    if predicted_plot_flag:
        predicted_plots(im_width2, im_height2, pipe, X_full2, y_full2)


# We can look at a third scene here.
if third_scene:
    print("\nHere we try our model for the whole third scene:")
    
    predicted3 = pipe.predict(X_full3)
    
    print(f"Classification report for classifier on third scene:\n"
          f"{metrics.classification_report(y_full3, predicted3)}\n")
    
    print("Model score on third scene:", pipe.score(X_full3, y_full3))
    
    disp1 = metrics.plot_confusion_matrix(pipe, X_full3, y_full3)
    disp1.figure_.suptitle("Confusion Matrix for Third Scene")
    plt.show()
    
    
    # Here we can plot a graph of the predicted vs. true ash points for this
    # extra scene.
    im_width3, im_height3 = right_column3-left_column3, bottom_row3-top_row3
    if predicted_plot_flag:
        predicted_plots(im_width3, im_height3, pipe, X_full3, y_full3)


# We can look at a fourth scene here.
if fourth_scene:
    print("\nHere we try our model for the whole fourth scene:")
    
    predicted4 = pipe.predict(X_full4)
    
    print(f"Classification report for classifier on fourth scene:\n"
          f"{metrics.classification_report(y_full4, predicted4)}\n")
    
    print("Model score on fourth scene:", pipe.score(X_full4, y_full4))
    
    disp4 = metrics.plot_confusion_matrix(pipe, X_full4, y_full4)
    disp4.figure_.suptitle("Confusion Matrix for Fourth Scene")
    plt.show()
    
    
    # Here we can plot a graph of the predicted vs. true ash points for this
    # extra scene.
    im_width4, im_height4 = right_column4-left_column4, bottom_row4-top_row4
    if predicted_plot_flag:
        predicted_plots(im_width4, im_height4, pipe, X_full4, y_full4)



# Here we can make plots of the denoised images
if denoised_image_flag:
    denoised_image(pipe, X_full, y_full, im_height, im_width, kernel_size=3)
    if second_scene:
        denoised_image(pipe, X_full2, y_full2, im_height2, im_width2, kernel_size=3)
    if third_scene:
        denoised_image(pipe, X_full3, y_full3, im_height3, im_width3, kernel_size=3)
    if fourth_scene:
        denoised_image(pipe, X_full4, y_full4, im_height4, im_width4, kernel_size=3)




#duration = 700  # milliseconds
#freq = 440  # Hz
#winsound.Beep(freq, duration)
print("\nRuntime:", datetime.now()-start)
sys.exit()

#############################################################################
############################# TRUE COLOUR IMAGE #############################
#############################################################################
# Here we can plot the true colour images for a scene
from pyresample import create_area_def

# First scene
observation_time = datetime(2020, 8, 1, 5)
# path_sat = '../data/himawari8/20200801_0500/'
path_sat = '../data/himawari8/'
# # Second scene
# observation_time = datetime(2020, 8, 1, 12)
# path_sat = '../data/himawari8/20200801_1200/'
# # Third scene
# observation_time = datetime(2020, 7, 30, 16)
# path_sat = '../data/himawari8/20200730_1600/'
# # Fourth scene
# observation_time = datetime(2020, 7, 28, 5)
# path_sat = '../data/himawari8/20200728_0500/'

filelist = glob.glob(path_sat + "*" + observation_time.strftime("%Y%m%d_%H%M") + "*.DAT")

scn = Scene(filelist, reader='ahi_hsd')
scn.available_dataset_names()
scn.load(['true_color'])
area_def = create_area_def('eqc', "+proj=eqc", units="degrees",
                           area_extent=(130, 20, 152, 32), resolution=0.02)
lcn = scn.resample(area_def)

lcn.load(['true_color'])
lcn.show('true_color')

sys.exit()



#############################################################################
####################### OTHER MODELS I'VE PLAYED WITH #######################
#############################################################################

# Here is the second one. We use a train/test split and scale the data to a
# Gaussian around zero to create a non-linear Support Vector Classifier.
print("\nSecond Attempt (Train/Test Split SVC):")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

num_transformer = Pipeline(steps=[('scaler', StandardScaler())])
# Not sure if I should use a column transformer to transform the data for each
# band separately? I think the standard scaler might already do columns
# separately.

svc_clf1 = Pipeline(steps=[('preprocessor', num_transformer), ('classifier',
                                svm.SVC(random_state=1))])
svc_clf1.fit(X_train, y_train)

predicted = svc_clf1.predict(X_test)
print(f"Classification report for classifier {svc_clf1}:\n"
      f"{metrics.classification_report(y_test, predicted)}\n")
disp2 = metrics.plot_confusion_matrix(svc_clf1, X_test, y_test)
disp2.figure_.suptitle("Confusion Matrix 2")
plt.show()
print("Model score:", svc_clf1.score(X_test, y_test))


# Here is the third one.
print("\nThird Attempt:")

params = dict()
params['classifier__C'] = (1.0, 1e6, 'log-uniform')
params['classifier__gamma'] = (1e-3, 100.0, 'log-uniform')
params['classifier__degree'] = (1, 5)
params['classifier__kernel'] = ['linear', 'poly', 'rbf', 'sigmoid']


num_transformer3 = Pipeline(steps=[('scaler', StandardScaler())])
svc_clf3 = Pipeline(steps=[('preprocessor', num_transformer3), ('classifier',
                            svm.SVC(random_state=1))])

cv1 = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

search = BayesSearchCV(estimator=svc_clf3, search_spaces=params, iid=True,
                       n_jobs=-1, cv=cv1)
search.fit(X, y)
print(search.best_score_) # 0.9864896479120365
print(search.best_params_) # OrderedDict([('classifier__C', 100.0),
                           # ('classifier__degree', 4),
                           # ('classifier__gamma', 0.1602895882738468),
                           # ('classifier__kernel', 'rbf')]) Note C hit it's
                           # max value of 100 so I raised the range of values
                           # to search over.


# Let's use the best parameters.
print("\nThis should be the best:")

num_transformer_best = Pipeline(steps=[('scaler', StandardScaler())])
svc_clf_best = Pipeline(steps=[('preprocessor', num_transformer_best),
                               ('classifier', svm.SVC(C=10000.0, gamma = 0.16,
                                                      degree=4, kernel='rbf',
                                                      random_state=1))])
cv_best = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

best_scores = cross_val_score(svc_clf_best, X, y, scoring='accuracy',
                              cv=cv_best, n_jobs=-1, error_score='raise')

print('Accuracy: %.3f (%.3f)' % (np.mean(best_scores), np.std(best_scores)))






# Prints the code runtime
print("\nRuntime:", datetime.now()-start)
