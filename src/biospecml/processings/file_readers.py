import os
import scipy.io as sio
import numpy as np


def read_mat(filename):
    """
    Function to read matlab file from OPUS FTIR Bruker.
    Code from Syahril, modified by Rafsanjani.
    
    Args:
    - filename (str): filename and directory location       
    
    Return:
    - w (int): Width of an image.
    - h (int): Height of an image.
    - p (array): Image data p(wavenmber,h,w).
    - wavenumber(array): Wavenumber arary (ex. 800-1000).
    - sp: Image data sp(wavenumber,w*h).
    """
  
    me, file_extension = os.path.splitext(filename)
    if (file_extension == '.dms'):
        print(f'{file_extension} not yet supported')
        # w, h, p, wavenumbers, sp = agiltooct(filename) 
        # return  w, h, p, wavenumbers, sp
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


def convert_mat_oct(sp, wn, tranpose_data:bool=False, key_name:str='ab', fname:str=None,
                    return_array:bool=False
                    ):
    """
    Convert spectral array from mat files to OCTAVVS friendly format by inserting
    wavenumber value in the first postion of spectral data.

    Args:
    - sp (np.ndarray): is the spectral array that has (x, y) shape
        - where x=features or wn, y=spectral data
    - wn (np.ndarray): flatten wavenumber i.e: array([980, 982, ...])
    - key_name (str): key name in the final converted mat file
    - fname (str): filename to save files

    """
    if tranpose_data: sp = sp.T
    
    new_sp = np.empty((sp.shape[0], sp.shape[1]+1)) # +1 for wn

    for i, w in enumerate(wn):
        sp_i = sp[i]
        sp_i = np.insert(sp_i, 0, w)
        new_sp[i] = sp_i
    
    if isinstance(fname, str):
        mat_arrays = {key_name: new_sp}
        sio.savemat(fname, mat_arrays)
    
    if return_array:
        return new_sp