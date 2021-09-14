# VAAC Polygons, Himawari-8 Data & Machine Learning Models for Identifying Volcanic Ash
Python notebook for plotting Tokyo Volcanic Ash Advisory (VAAC) polygons over Advanced Himawari Imager (AHI) data and creating data sets used to test and train machine learning algorithms for detecting volcanic ash clouds from the 2020 Nishinoshima eruption.

This repository contains the code required to reproduce results found by M. Waugh and A. Prata over the summer of 2021 when investigating the effectiveness of supervised machine learning algorithms at using AHI data to classify volcanic ash clouds. 

The main body of code is included as 'MainCode.py', which calls functions from 'Functions1.py' to train, test and visualise the results of machine learning models. Emphasis is given to Linear Kernel Support Vector Classifier (SVC) models from the scitkit-learn python package which proved fast to implement over the size of data sets used and robustly provided results accurate from a qualitative and quantitative standpoint.

## Data
Data is taken from a HTML file from the Tokyo VAAC containing vertex positions of the volcanic ash cloud polygon, which can be plotted on a map to create a graphic showing a coarse human-produced estimate of the cloud shape and position given by the VAAC.
![Example Volcanic Ash Graphic](/figures/ex_VAG.jpg)
A corrected brightness temperature difference can be calculated using bands 14 and 15 from the AHI to represent a finer grained estimate of volcanic ash position.
![Example Corrected Brightness Temperature Difference Representation](/figures/ex_BTD.jpg)
A final data set is created by taking all ash pixels with a negative corrected brightness temperature difference inside the VAAC polygon to contain ash and all others to contain no ash. This combines the human VAAC polygon classification method with the automated corrected brightness temperature method to produce a blended data set of pixels labelled as ash or not ash.
![Example Data Set](/figures/ex_dataset.jpg)

## Model Creation and Training
Models are trained pixel by pixel using the data sets created by being given all 16 bands of the raw AHI data alongside the corresponding label of ash or not ash. The amount and variety of data required to make accurate predictions was being investigated so has been left flexible in 'MainCode.py'. Linear kernel SVCs were particularly interesting to investigate because the model coefficients signify the importances of each AHI band which could be compared to previously known results.
![Example Linear Kernel Support Vector Classifier Coefficients](/figures/ex_coefficients.jpg)

## Model Predictions
Quantitative and qualitative measures of algorithm performance can be made from 'MainCode.py'. Qualitatively, graphics can be produced showing ash and not ash pixels when given AHI data from a frame. Noise reduction can also be carried out, which removes small regions designated as ash.
![Example Predictions Given by a Linear Kernel Support Vector Classifier Run Over An AHI Segment](/figures/ex_prediction.jpg)
