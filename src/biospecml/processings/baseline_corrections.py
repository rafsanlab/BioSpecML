from pybaselines import Baseline
from scipy.spatial import QhullError
import numpy as np

def baseline_corrections(sp, wn, method='rubberband', lim=None,
                         return_baselines:bool=False,
                         replace_arr:bool=True, replace_values=None):
    
    sp = sp.T
    sp_corr = np.empty_like(sp)
    sp_base = np.empty_like(sp)
    baseline_fitter = Baseline(x_data=wn)
    err_idx_lst = []

    n = 0
    for i, spectra in enumerate(sp):
        n+=1
        try:
            if method=='rubberband':
                bkg_1, params_1 = baseline_fitter.rubberband(spectra)
            else:
                raise Exception('Will support more baseline correction method upon request.')
            spectra_corr = spectra - bkg_1
            sp_corr[i] = spectra_corr
            sp_base[i] = bkg_1
        except (IndexError, QhullError):
            err_idx_lst.append(i)
            if replace_arr:
                if replace_values is None:
                    replace_values = (sp_corr[i-1] + sp_corr[i-2])/2
                sp_corr[i] = replace_values

        if lim is not None and n >= lim:
            break
    
    if len(err_idx_lst)>0:
        print(f'Total data with errors : {len(err_idx_lst)}')

    if return_baselines:
        return sp_corr.T, err_idx_lst, sp_base.T
    else:
        return sp_corr.T, err_idx_lst