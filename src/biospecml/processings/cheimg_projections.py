import numpy as np

def projection_area(p:np.ndarray, wavenumbers:np.ndarray):
    """
    FTIR Image Reconstruction where Pixel Intensity = Area Under Curve of SPECTRUM
    Code by Syahril, modified by Rafsanjani.

    Args:
    - sp (array): (wavenumber,h,w)
    - wavenumbers (array): wavenumbers
    
    Return:
    - cc (array): h x w image projection
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
    - p (np.ndarray) : the datacube of ftir image

    Returns:
    - img_std(np.ndarray) : image projection

  """
    img_std = np.zeros((p.shape[1],p.shape[2]))
    for i in range(p.shape[1]):
        for j in range(p.shape[2]):
            img_std[i,j] = np.std(p[:,i,j])
    return img_std
