import scipy.io as sio
import numpy as np
import os


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
