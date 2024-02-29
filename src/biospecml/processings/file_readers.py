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