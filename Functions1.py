# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 11:34:11 2021

@author: mwaugh98

File contains functions used in Trial2.py to create datasets, model ML 
algorithms and evaluate algorithm performance.
"""

import codecs
from bs4 import BeautifulSoup
import numpy as np
from satpy import Scene
import glob
from matplotlib.path import Path
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import sys
import cv2 as cv




def poly_latlon(polygon_filename, polygon_fileloc, create_VAG=False):
    """
    Extracts polygon vertex locations from html Tokyo VAAC file.
    """
    
    # Read html VAAC data into a python list using bs4 so we can extract
    # polygon vertices
    with codecs.open(polygon_fileloc + polygon_filename, 'r') as f: # codecs might not be required here, or rather io might be better?
        html = f.read() # This sets the variable html as the contents of fn_vacc (rather than the file itself)
    soup = BeautifulSoup(html, features="html.parser") # This gives us the BeautifulSoup-ifyed version (Pythonic?), read by default parser (can't get lxml to work?)
    div = soup.find('div') # This extracts the main bit of text from soup (ignoring the main title and lines and stuff)
    vaac = [str(s) for s in div.contents if str(s) != '<br/>' and str(s) != str('\n')]
    
    # Important information is contained in list entries 14 and 15
    polygon = vaac[14] + vaac[15]
    coords = polygon[22:-11].split('-') # Gets rid of unwanted information from polygon and creates a list of each vertex
    
    # Convert VAAC coords to desired form
    lons_polygon = []
    lats_polygon = []
    
    for coord in coords:
        lat_str, lon_str = coord.strip().split(' ') # Strips extra blank spaces from element in coords, then splits to lat & lon
        lat_decimal, lon_decimal = float(lat_str[1:]) / 100., float(lon_str[1:]) / 100.
        lat_split = '{0:.2f}'.format(lat_decimal).split('.') # Splits our decimal to list [whole number, 2 sf of decimal part]
        lon_split = '{0:.2f}'.format(lon_decimal).split('.')
        lat_float = float(lat_split[0]) + float(lat_split[1]) / 60. # Changing our coordinates to different units - maybe arcsec?
        lon_float = float(lon_split[0]) + float(lon_split[1]) / 60.
        if "E" in lon_str: # This sets eastwards points as positive and westwards points as negative in lons_polygon
            lons_polygon.append(lon_float)
        else:
            lons_polygon.append(-lon_float)
        if "N" in lat_str: # This sets northwards points as positive and southwards points as negative in lats_polygon
            lats_polygon.append(lat_float)
        else:
            lats_polygon.append(-lat_float)
    
    lons_polygon.append(lons_polygon[0]) # Adds the first point to the end of the list, duplicating it
    lats_polygon.append(lats_polygon[0])
    
    if create_VAG:
        produce_VAG(polygon_filename, vaac, lats_polygon, lons_polygon)

    return lats_polygon, lons_polygon



def BTDs(B14, B15, lats_polygon, lons_polygon, lcn, r1, r2, c1, c2):
    """
    Calculates brightness temperatures corrected by the Yu et al. 
    semi-empirical water vapour correction. Then masks pixels outside the 
    Tokyo VAAC polygon.
    """
    # We now use the satellite data and VAAC polygon to determine ash location.
    # Plot brightness temperature difference to detect ash (-ve = ash, +ve = water/ice)
    T11 = B14
    T12 = B15
    BTD = T11 - T12
    
    # Notice how the whole plume isn't detected by this difference. This is due to 
    # the interference of water vapour. To address this we can apply the Yu et al. 
    # semi-emprical water vapour correction.
    # Note: increasing b reduces the water vapour correction)
    b = 4.5
    wv = np.exp(6. * (T11 / 320.) - b)
    BTD_wv = BTD - wv
    
    
    # We here mask the data outside the VAAC polygon
    # Creates list of tuples (lon, lat) of polygon coordinates
    polygon = [(lon, lat) for lon, lat in zip(lons_polygon, lats_polygon)]
    
    # Get lons and lats for each point in our area? Not sure if these are correct 
    # or if I need to convert them? They seem to work...?
    lons, lats = lcn['B14'].attrs['area'].get_lonlats()
    lons, lats = lons[r1:r2, c1:c2], lats[r1:r2, c1:c2]
    
    # Create the array of BTDs for just ash
    x, y = lons.flatten(), lats.flatten()
    points = np.vstack((x,y)).T # Creates stack of (repeating) lon and lat pairs
    p = Path(polygon) # Connects polygon coords to make closed polygon
    grid = p.contains_points(points) # Returns whether closed polygon contains point as array (example: [False False False ... False False False])
    Mask = grid.reshape(lats.shape) # Creates a mask of the shape of our image (BTD_wv) with each index describing if the point is inside the closed polygon or not
    BTD_wv_ma = np.ma.array(BTD_wv, mask=Mask==False) # BTDs inside polygon only. mask=Mask==False sets all the points of BTD_wv which have a False value in Mask to be masked
    
    # Masks BTDs greater than or equal to zero
    BTD_wv_ma = np.ma.masked_greater_equal(BTD_wv_ma, 0)
    
    return BTD_wv, BTD_wv_ma



def dataset(polygon_filename, polygon_fileloc, sat_datetime, sat_location, 
            top_row, bottom_row, left_column, right_column, 
            desired_channels = {'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 
                                'B07', 'B08','B09', 'B10', 'B11', 'B12', 
                                'B13','B14', 'B15', 'B16'}, 
            create_VAG=False, plot_BTDs=False, plot_ash=False):
    """
    Uses Tokyo VAAC polygon and brightness temperatures calculated from the
    Himawari-8 satellite to determine whether a pixel contains ash or not then
    creates outputs containg this information alongside chosen Himawari-8 data.
    
    Input:
        polygon_filename - Name of html Tokyo VAAC file.
        polygon_location - Path from current folder to polygon html file.
        sat_datetime - datetime object stating date and time satellite data
                       was gathered.
        sat_location - Path from current folder to satellite data file.
        top_row - The upper boundary of the region being considered.
        bottom_row - The lower boundary of the region being considered.
        left_column - The left-most boundary of the region being considered.
        right_column - The right-most boundary of the region being considered.
        desired_channels - Set containing desired Himawari-8 channels. 
                           Defaults to all channels.
        create_VAG - Produces a Volcanic Ash Graphic (VAG) showing the VAAC
                     polygon.
        plot_BTDs - Produces a plot of the brightness temperatures 
                    corrected by the Yu et al. semi-empirical water vapour 
                    correction.
                           
    Return:
        X - Design matrix X. Each row represents a pixel and each column
            represents the data from each available band. Example: if all 
            bands are available then X shall be of the form:
                [[Pixel_1(B01) ... Pixel_1(B16)],
                 [Pixel_2(B01) ... Pixel_2(B16)],
                 ...
                 [Pixel_N(B01) ... Pixel_N(B16)]]
        y - 1D array of labels y. Contains 1 for pixels containing ash and 0
            for those that do not.
        ash_num - The number of data points that contain ash.
    """
    # Get polygon vertex locations using poly_latlon()
    lats_polygon, lons_polygon = poly_latlon(polygon_filename, polygon_fileloc,
                                             create_VAG)
    
    # We now wish to extract all the desired satellite data we can.
    # Generate list of filenames for each Himawari band by finding each data 
    # file for our desired date & time.
    filelist = glob.glob(sat_location + "*" +
                         sat_datetime.strftime("%Y%m%d_%H%M") + "*.DAT")
    
    scn = Scene(filelist, reader='ahi_hsd')
    
    # Create set of which channels we have data for
    data_channels = set()
    for i in scn.available_dataset_names():
        data_channels.add(i)   
    
    # Create list of desired channels that are available and optionally order
    # it. B14 and B15 have to be in this list by construction of the rest of 
    # the code so check if they are there. This doesn't mean they have to be
    # in X, we need these channels for the BTDs anyway.
    available_channels = list(data_channels & desired_channels)
    if 'B14' not in available_channels:
        available_channels.append('B14')
    if 'B15' not in available_channels:
        available_channels.append('B15')
    available_channels.sort()
    
    # Load in available desired channels
    scn.load(available_channels)

    # Resample our data to the same resolution (that of our coarsest band).
    lcn = scn.resample(scn.coarsest_area(), resampler='native')
    
    # Creates column arrays containing the satellite data for all bands that 
    # exist. Must be a cleaner way of doing this?
    if "B01" in available_channels:
        B01 = lcn['B01'].values[top_row:bottom_row, left_column:right_column].reshape(-1,1)
    if "B02" in available_channels:
        B02 = lcn['B02'].values[top_row:bottom_row, left_column:right_column].reshape(-1,1)
    if "B03" in available_channels:
        B03 = lcn['B03'].values[top_row:bottom_row, left_column:right_column].reshape(-1,1)
    if "B04" in available_channels:
        B04 = lcn['B04'].values[top_row:bottom_row, left_column:right_column].reshape(-1,1)
    if "B05" in available_channels:
        B05 = lcn['B05'].values[top_row:bottom_row, left_column:right_column].reshape(-1,1)
    if "B06" in available_channels:
        B06 = lcn['B06'].values[top_row:bottom_row, left_column:right_column].reshape(-1,1)
    if "B07" in available_channels:
        B07 = lcn['B07'].values[top_row:bottom_row, left_column:right_column].reshape(-1,1)
    if "B08" in available_channels:
        B08 = lcn['B08'].values[top_row:bottom_row, left_column:right_column].reshape(-1,1)
    if "B09" in available_channels:
        B09 = lcn['B09'].values[top_row:bottom_row, left_column:right_column].reshape(-1,1)
    if "B10" in available_channels:
        B10 = lcn['B10'].values[top_row:bottom_row, left_column:right_column].reshape(-1,1)
    if "B11" in available_channels:
        B11 = lcn['B11'].values[top_row:bottom_row, left_column:right_column].reshape(-1,1)
    if "B12" in available_channels:
        B12 = lcn['B12'].values[top_row:bottom_row, left_column:right_column].reshape(-1,1)
    if "B13" in available_channels:
        B13 = lcn['B13'].values[top_row:bottom_row, left_column:right_column].reshape(-1,1)
    if "B16" in available_channels:
        B16 = lcn['B16'].values[top_row:bottom_row, left_column:right_column].reshape(-1,1)
    
    # These channels require the 'else' statement otherwise we run into errors
    if "B14" in data_channels:
        B14 = lcn['B14'].values[top_row:bottom_row, left_column:right_column].reshape(-1,1)
    else:
        print("Brightness temperature difference cannot be calculated. Exiting code.")
        sys.exit()
    if "B15" in data_channels:
        B15 = lcn['B15'].values[top_row:bottom_row, left_column:right_column].reshape(-1,1)
    else:
        print("Brightness temperature difference cannot be calculated. Exiting code.")
        sys.exit()
    
    
    # Get corrected (and masked corrected) BTDs with function BTDs().
    BTD_wv, BTD_wv_ma = BTDs(B14, B15, lats_polygon, lons_polygon, lcn, 
                             top_row, bottom_row, left_column, right_column)

    # Plot corrected BTDs if desired.
    if plot_BTDs:
        plot_BTD_graph(BTD_wv, bottom_row, top_row, left_column, right_column,
                       masked=False)
    
    # Plot masked corrected BTDs if desired.
    if plot_ash:
        plot_BTD_graph(BTD_wv_ma, bottom_row, top_row, left_column,
                       right_column, masked=True)

    
    # Now we collate the data to create X and y.
    # Turn B14 & B15 into columns like the other channels to create X
    B14 = B14.reshape(-1,1)
    B15 = B15.reshape(-1,1)
    
    # Creates column of zeros with as many rows as there are pixels.
    X = np.empty(B14.shape)
    
    # Adds each band data to X. Must be a cleaner way of doing this?
    if "B01" in available_channels:
        X = np.hstack((X, B01))
    if "B02" in available_channels:
        X = np.hstack((X, B02))
    if "B03" in available_channels:
        X = np.hstack((X, B03))
    if "B04" in available_channels:
        X = np.hstack((X, B04))
    if "B05" in available_channels:
        X = np.hstack((X, B05))
    if "B06" in available_channels:
        X = np.hstack((X, B06))
    if "B07" in available_channels:
        X = np.hstack((X, B07))
    if "B08" in available_channels:
        X = np.hstack((X, B08))
    if "B09" in available_channels:
        X = np.hstack((X, B09))
    if "B10" in available_channels:
        X = np.hstack((X, B10))
    if "B11" in available_channels:
        X = np.hstack((X, B11))
    if "B12" in available_channels:
        X = np.hstack((X, B12))
    if "B13" in available_channels:
        X = np.hstack((X, B13))
    if "B14" in desired_channels:
        X = np.hstack((X, B14))
    if "B15" in desired_channels:
        X = np.hstack((X, B15))
    if "B16" in available_channels:
        X = np.hstack((X, B16))
    
    X = np.delete(X, 0, axis=1)
    
    
    # Now create y by replacing masked BTD values with zero then setting all 
    # values less than zero to 1 (ash) and all more than zero to 0 (not ash) 
    # before reshaping
    y = BTD_wv_ma.filled(0)
    y = np.where(y < 0, 1, 0).ravel()
    
    # Create dictionary showing number of occurences each of 0 and 1 in y
    unique, counts = np.unique(y, return_counts=True)
    counts_dict = dict(zip(unique, counts))
    
    # Gives number of datapoints that contain ash (i.e. number of unmasked 
    # datapoints)
    ash_num = len(BTD_wv_ma.compressed())
    # tot_ash_num = len(BTD_wv_ash.compressed())
    # data_threshold = ash_num/tot_ash_num # Got rid of this because I don't think it's that useful
    tot_num = len(y)
    
    # Does a quick check that the number of 'ash' points in y matches the 
    # number of BTDs less than zero. (I'm fairly sure this has to be true by 
    # the code's construction but settles my mind to check). If all fine then 
    # prints data.
    if counts_dict[1] == ash_num:
        print("Available channels:\n", available_channels)
        print("Total number of datapoints:", tot_num)
        return X, y, ash_num
    else:
        print("Whoops, it looks like something has gone wrong!")
        sys.exit()



def reduce_dataset(X_full, y_full, ash_num, ratio=1):
    """    
    Reduces the size of the dataset so that we have equal numbers of 'not ash'
    and 'ash' data points. The function takes all the 'ash' points and an
    equal number of randomly selected of 'not ash' points to create the 
    reduced dataset. Note: the ratio of 'not ash' to 'ash' points can be
    altered to affect the behaviour of the algorithm.

    Parameters
    ----------
    X_full : The full matrix of parameters for each pixel we want to reduce.
    y_full : The full classification vector corresponding to X.
    ash_num : The number of ash pixels in the dataset we would like to reduce.
    ratio : The ratio of 'not ash' to 'ash' pixels in the reduced dataset.
    
    Returns
    -------
    X_reduced : The reduced matrix of parameters for each pixel.
    y_reduced : The reduced classification vector corresponding to X_reduced.

    """
    not_ash_ind = np.where(y_full==0)[0]
    ash_ind = np.where(y_full==1)[0]
    not_ash_num = ratio*ash_num
    reduced_not_ash_ind = np.random.choice(not_ash_ind, not_ash_num)
    reduced_ind = np.concatenate((ash_ind, reduced_not_ash_ind))
    
    X_reduced = np.array(X_full[reduced_ind])
    y_reduced = np.array(y_full[reduced_ind])
    return X_reduced, y_reduced



def add_weird_region(X_full, y_full, X, y):
    i=1
    X_left, X_right = 20080, 20200
    while i < 90:
        i = i+1
        X_left, X_right = X_left+400, X_right+400
        X_add = np.array(X_full[X_left:X_right])
        y_add = np.array(y_full[X_left:X_right])
        X = np.concatenate((X, X_add))
        y = np.concatenate((y, y_add))
    return X, y



def produce_VAG(fn_vacc, vaac, lats_polygon, lons_polygon):
    """
    Produces a Volcanic Ash Graphic (VAG) of the VAAC polygon.

    Parameters
    ----------
    fn_vacc : Filename for polygon html file.
    vaac : Cannot remember exactly what this is but it is just the time and
           date or something.
    lats_polygon : Polygon vertex latitudes.
    lons_polygon : Polygon vertex longitudes.

    Returns
    -------
    None. Should produce a Volcanic Ash Graphic (VAG).

    """
    # Volcano position. Could be found from file but just stated here for 
    # convenience.
    volc_lon = 140.874
    volc_lat = 27.2471
    
    # Creates empty figure and sets background colour to white.
    fig = plt.figure(figsize=(12, 6))
    fig.set_facecolor('w')
    
    # Adds an axes to the figure in the Plate Carree coordinate system (or 
    # projection).
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Adds a title to our axes with the DTG given.
    DTG = vaac[3]
    ax.set_title(label='Nishinoshima Volcanic Ash Graphic (VAG) from the ' +
                 'Himawari-8 Satellite\n' + DTG)
    
    # This sets the lat-lon range over which our map shall show then adds the 
    # coastlines and gridlines.
    plot_extent = [120, 160, 15, 35] #[min_lon, max_lon, min_lat, max_lat]
    ax.set_extent(plot_extent)
    ax.coastlines(resolution='10m', color='g')
    ax.gridlines(draw_labels=True, dms=True)
    
    # This creates a marker for Nishinoshima and labels it.
    ax.plot(volc_lon, volc_lat, marker='^', color='red', markersize=8)
    ax.text(volc_lon + 0.3, volc_lat, 'Nishinoshima', 
            verticalalignment='bottom', horizontalalignment='left')
    
    # Plots our polygon over the map.
    ax.plot(lons_polygon, lats_polygon, '-k')
    
    # Saves the image in the data folder.
    png_name = fn_vacc[:-9] + 'VAG'
    plt.savefig('../vaac_polygons_sat/data/vaa/' + png_name)
    plt.show()



def plot_BTD_graph(BTDs, bottom_row, top_row, left_column, right_column,
                   masked=False):
    """
    Produces plots of brightness temperature differences for either masked or
    unmasked arrays.
    
    """
    im_height, im_width = bottom_row-top_row, right_column-left_column
    BTD_wv_plot = BTDs.reshape(im_height, im_width)
    
    fig, axes = plt.subplots(figsize=(18,8))
    if masked:
        axes.set_title('Masked Corrected Brightness Temperature Difference', 
                       fontweight='bold')
        cmap_col = plt.get_cmap('gist_heat')
    else:
        axes.set_title('Corrected Brightness Temperature Difference', 
                       fontweight='bold')
        cmap_col = plt.get_cmap('RdBu')
    axes.imshow(BTD_wv_plot, cmap=cmap_col, interpolation='nearest')
    axes.xaxis.set_visible(False)
    axes.yaxis.set_visible(False)



def predicted_plots(im_width, im_height, pipe, X_full, y_full, opening=False):
    """
    Creates a plot of the predictions of pipe over the input image described
    by X_full, as well as the 'true' image according to y_full alongside 
    plots describing the points missed by pipe and the additional points
    predicted by pipe.

    Parameters
    ----------
    im_width : Width of image in pixels.
    im_height : Height of image in pixels.
    pipe : Pipeline containing classifier to produce the plot of predicted 
           ash points.
    X_full : Full matrix of parameters for each pixel that we want to create
             our image over.
    y_full : Full classification vector corresponding to X_full.
    opening : Boolean input for if the image should be opened or not using
              OpenCV.

    Returns
    -------
    None.

    """
    # First create our shaped arrays of ash and not ash pixels
    tot_predicted_plot = pipe.predict(X_full).reshape(im_height, im_width)
    
    tot_true_plot = y_full.reshape(im_height, im_width)
    
    tot_additional_plot = tot_predicted_plot - tot_true_plot
    tot_additional_plot[tot_additional_plot < 0] = 0
    tot_additional_plot.reshape(im_height, im_width)
    
    tot_missed_plot = -tot_predicted_plot + tot_true_plot
    tot_missed_plot[tot_missed_plot < 0] = 0
    tot_missed_plot.reshape(im_height, im_width)
    
    # We can now create the plots
    cmap_col = plt.get_cmap('Greys')
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(11,8))
    fig.suptitle('Comparing Ash Pixels from SVC Against Brightness '+
                 'Temperature Difference & VAAC Polygon Method', 
                 fontweight='bold')
    axes[0,0].imshow(tot_predicted_plot, cmap=cmap_col, interpolation='nearest')
    axes[0,0].set_title('SVC Ash Prediction')
    axes[0,0].xaxis.set_visible(False)
    axes[0,0].yaxis.set_visible(False)
    
    axes[0,1].imshow(tot_true_plot, cmap=cmap_col, interpolation='nearest')
    axes[0,1].set_title('Brightness Temperature Difference & VAAC Polygon Prediction')
    axes[0,1].xaxis.set_visible(False)
    axes[0,1].yaxis.set_visible(False)
    
    axes[1,0].imshow(tot_additional_plot, cmap=cmap_col, interpolation='nearest')
    axes[1,0].set_title('Additional Ash Points Predicted by SVC')
    axes[1,0].xaxis.set_visible(False)
    axes[1,0].yaxis.set_visible(False)
    
    axes[1,1].imshow(tot_missed_plot, cmap=cmap_col, interpolation='nearest')
    axes[1,1].set_title('Ash Points Missed by SVC')
    axes[1,1].xaxis.set_visible(False)
    axes[1,1].yaxis.set_visible(False)



def denoised_image(pipe, X_full, y_full, im_height, im_width, kernel_size=3):
    """
    Produces a figure containing the BTD and VAAC polygon ash prediction,
    machine learning (ML) model prediction and the denoised ML model 
    prediction.

    Parameters
    ----------
    pipe : ML model.
    X_full : Feature matrix for scene.
    y_full : Label vector for scene.
    im_height : Pixel height of scene.
    im_width : Pixel width of scene.
    kernel_size : Integer. Dimension of square kernel. The default is 3.

    Returns
    -------
    None. Should output a figure.

    """
    cmap_col='Greys'
    kernel = np.ones((kernel_size,kernel_size), np.uint8)

    tot_predicted_plot = pipe.predict(X_full).reshape(im_height, im_width).astype('uint8')
    tot_true_plot = y_full.reshape(im_height, im_width)
    opened_predicted_plot = cv.morphologyEx(tot_predicted_plot, cv.MORPH_OPEN, 
                                            kernel)

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(30,7))
    fig.suptitle('Comparing Ash Pixels from SVC Against Brightness '+
                 'Temperature Difference & VAAC Polygon Method', 
                 fontweight='bold', fontsize=15)
    
    axes[0].imshow(tot_true_plot, cmap=cmap_col, interpolation='nearest')
    axes[0].set_title('Brightness Temperature Difference & VAAC Polygon Prediction', fontsize=14)
    axes[0].xaxis.set_visible(False)
    axes[0].yaxis.set_visible(False)
    axes[1].imshow(tot_predicted_plot, cmap=cmap_col, interpolation='nearest')
    axes[1].set_title('Model Prediction', fontsize=14)
    axes[1].xaxis.set_visible(False)
    axes[1].yaxis.set_visible(False)
    axes[2].imshow(opened_predicted_plot, cmap=cmap_col, interpolation='nearest')
    axes[2].set_title('Model Prediction with Noise Removed', fontsize=14)
    axes[2].xaxis.set_visible(False)
    axes[2].yaxis.set_visible(False)



def coefficients_plot(pipe, features_names=[]):
    """
    Creates a plot of the coefficients of a (linear) SVC.

    Parameters
    ----------
    pipe : Pipeline containing a step "classifier" which is a (linear) SVC.
    features_names : List of the names of each feature each coefficient
                     corresponds to.

    Returns
    -------
    None. Should create a plot of the coefficients of the (linear) SVC.

    """
    importances = pipe['classifier'].coef_
    # Note this naming doesn't work if we don't have all the desired channels.
    # This isn't normally a problem for me so I'll just leave this as it is
    # right now.
    if features_names==[]:
        features_names = ['B01 (455 nm)', 'B02 (510 nm)', 'B03 (645 nm)', 
                          'B04 (860 nm)', 'B05 (1610 nm)', 'B06 (2260 nm)', 
                          'B07 (3.85 $\mu$m)', 'B08 (6.25 $\mu$m)', 
                          'B09 (6.95 $\mu$m)', 'B10 (7.35 $\mu$m)', 
                          'B11 (8.60 $\mu$m)', 'B12 (9.63 $\mu$m)', 
                          'B13 (10.45 $\mu$m)', 'B14 (11.20 $\mu$m)', 
                          'B15 (12.35 $\mu$m)', 'B16 (13.30 $\mu$m)']

    importances, features_names = zip(*sorted(zip(importances[0], 
                                                  features_names)))
    plt.barh(range(len(features_names)), importances)
    plt.yticks(range(len(features_names)), features_names)
    plt.title('Linear SVC Coefficients')
    plt.show()
