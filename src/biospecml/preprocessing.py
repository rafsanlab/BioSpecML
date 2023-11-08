import scipy.io as sio
import numpy as np
import os
import cv2 as cv
import pandas as pd

def read_mat(filename):
    """
    Function to read matlab file from OPUS FTIR Bruker.
    Code from Syahril, modified by Rafsanjani.
    
    Args:
        filename(str): filename and directory location       
    Return:
        w(int): Width of an image.
        h(int): Height of an image.
        p(array): Image data p(wavenmber,h,w).
        wavenumber(array): Wavenumber arary (ex. 800-1000).
        sp: Image data sp(wavenumber,w*h).
    """
  
    me, file_extension = os.path.splitext(filename)
    if (file_extension == '.dms'):
        print(file_extension)
        w, h, p, wavenumbers, sp = agiltooct(filename) 
        return  w, h, p, wavenumbers, sp
    else:       
        s = sio.loadmat(filename)
        info = sio.whosmat(filename)
        ss = s[(str(info[0][0]))]
        wavenumber=ss[:,0]
        sizex = len(wavenumber)
        sizemx, sizemy = np.shape(ss)
        sp = ss[:,1:]
        if (len(info)) > 1:
            (l,*_) = s['wh']
            w , h = l
        else:     
            w = int(np.sqrt(sp.shape[1]))
            h = sp.shape[1] // w
            if w * h != sp.shape[1]:
                w = sp.shape[1]
                h = 1
        p = sp.reshape(sizex,h,w,order='C')
        return  w, h, p, wavenumber, sp


def projection_area(p:np.ndarray, wavenumbers:np.ndarray):
    """
    FTIR Image Reconstruction where Pixel Intensity = Area Under Curve of SPECTRUM
    Code by Syahril, modified by Rafsanjani.

    Args:
        sp(array): (wavenumber,h,w)
        wavenumbers(array): wavenumbers
    Return:
        ?(array): h x w image projection
    """

    i,j,k = np.shape(p)
    wavenumbers = np.sort(wavenumbers) #because data are scanned from high to low
    cc=np.zeros((j,k))
    for ii in range(0,j):
            for jj in range(0,k):
                cc[ii,jj]= np.trapz(p[:,ii,jj],wavenumbers)
    # cc = np.trapz(p[:,:,:],wavenumbers,axis=0)
    return cc


def projection_std(p:np.ndarray):
    """
    Apply projection based on standard deviation of p (from read_mat()).

    Args:
        p(np.ndarray) : the datacube of ftir image

    Returns:
        img_std(np.ndarray) : image projection

  """
    img_std = np.zeros((p.shape[1],p.shape[2]))
    for i in range(p.shape[1]):
        for j in range(p.shape[2]):
            img_std[i,j] = np.std(p[:,i,j])
    return img_std

def img_inverse(img:np.ndarray, point:tuple=(0,0), background:int=0):
    """
    Inverse image using cv.bitwise_not by checking the point given whether it fit
    the background. If it's not, it inverse will be run. (this func is use to invert
    K-means 1 and 0 cluster output results).

    Args:
        img : the K-means image projection (2D image)
        point : background coordinate
        background : background value (i.e; 1 or 0)

    Returns:
        img : inversed or not inversed image
    """
    if img[point[0], point[1]] != background:
        img = cv.bitwise_not(img)
        return img
    else:
        return img



def img_thres_otsu(img, blur_kernel=(3,3), tval=0, maxval=255):
    """
    Applies Otsu thresholding to an image.

    Args:
        img (numpy.ndarray): Array representing the input image.
        blur_kernel (tuple): Gaussian blur kernel size.
        tval (int): Thresholding value.
        maxval (int): Value to be assigned to thresholded pixels.

    Returns:
        numpy.ndarray: Thresholded image array.
    """
    img, blur_kernel, tval, maxval = img, blur_kernel, tval, maxval

    # checking img input
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    elif len(img.shape) == 2:
        img = img
    else:
        print('Error: Image input invalid.')
        return None
  
    # img -> blur -> Otsu
    img = cv.GaussianBlur(img, blur_kernel, 0)
    thresh = cv.threshold(img, tval, maxval, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    return thresh


def img_rm_debris(img, X1=0.01):
    """
    Remove small particles from a 2D image.

    Args:
        img (numpy.ndarray): Array representing the input 2D image.
        X1 (float): Multiplier of the average area of the image.
            Smaller X1 values remove larger particles.

    Returns:
        numpy.ndarray: Thresholded image array.
    """

    thresh, X1 = img, X1

    # checking img input
    if len(img.shape) != 2:
        print('Error: Image input invalid.')
        return None

    # determine average area
    average_area = [] 
    cnts = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x,y,w,h = cv.boundingRect(c)
        area = w * h
        average_area.append(area)
    average = sum(average_area) / len(average_area)

    # remove 'debris'
    cnts = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv.contourArea(c)
        if area < average * X1:
             cv.drawContours(thresh, [c], -1, (0,0,0), -1)
    return thresh


def img_rm_holes(img, X1=0.1, holes_kernel=(5,5), iterations=2):
    """
    Remove holes from an 2D image array.

    Args:
        img(np.ndarray): an array of 2D image.
        X1(float): multiplier of average area size.
        holes_kernel(tup): size of holes to be remove.
        interations(int): number of iterations .

    Returns:
        close(np.ndarray): image array.
    """
    thresh, X1, iterations = img, X1, iterations

    # checking img input
    if len(img.shape) != 2:
        print('Error: Image input invalid.')
        return None

    # determine average area
    average_area = [] 
    cnts = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x,y,w,h = cv.boundingRect(c)
        area = w * h
        average_area.append(area)
    average = sum(average_area) / len(average_area)

    # remove 'holes'
    cnts = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv.contourArea(c)
        if area < average * X1:
            cv.drawContours(thresh, [c], -1, (0,0,0), -1)
  
    # Morph close and invert image
    kernel = cv.getStructuringElement(cv.MORPH_RECT, holes_kernel)
    close = cv.morphologyEx(
        thresh,cv.MORPH_CLOSE,
        kernel, iterations=iterations
         )
  
    return close

def calc_snr(dfX, signal_range, noise_range):
    """
    Calculate signal to noise (SNR) ratio based on mean signal / std noise.

    Args:
        dfX(pd.DataFrame): matrix of a df.
        signal_range(tuple): wavenumber range from df columns.
        noise_range(tuple): wavenumer ranger from df columns.
    Return:
        SNR ratio(np.array).
    """

    numeric_cols = pd.to_numeric(dfX.columns) # convert columns to numbers
    signal = dfX.loc[:, (numeric_cols >= signal_range[1]) & (numeric_cols <= signal_range[0])]
    signal = signal.mean(axis=1) # calculate signal
    noise = dfX.loc[:, (numeric_cols >= noise_range[1]) & (numeric_cols <= noise_range[0])]
    noise = noise.std(axis=1) # calculate noise
    snr = np.where(noise==0, 0, signal/noise) # calculate SNR ratio
    return snr

def calc_outliers_threshold(snr_column, n):
    """
    Calculate the outliers threshold of all spectra dataframe based on the formula
    = mean + n * SD, use this value to filter spectra outliers from the dataset.

    Args:
        snr_column(pd.core.series.Series): SNR column.
        n(int): SD multiplier.
    Return:
        outliers_threshold(float): the threshold value.
    """

    outliers_threshold = snr_column.mean() + (n * snr_column.std()) # mean+3xSD
    return outliers_threshold