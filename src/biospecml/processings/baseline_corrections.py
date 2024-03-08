from pybaselines import Baseline
from scipy.spatial import QhullError
import numpy as np

def baseline_corrections(sp, wn, method='rubberband', lim=None, return_baselines:bool=False,
                         replace_arr:bool=True, replace_values=None):
    """
    Apply baseline correction to the sp data (wn, w * h).
    Option to replace_values if encounter IndexError or QhullError during the triangulation process,
    use an array that are has the same shape with wn. If None, will find the mean between the adjacent
    spectra and replace the error spectra if replace_arr=True. Will return err_idx_lst which is the index list
    where is the specra has errors.
    """
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
                spectra_corr = spectra - bkg_1
            else:
                raise Exception('Will support more baseline correction method upon request.')
       
        except (IndexError, QhullError):
            err_idx_lst.append(i)
            if replace_arr:
                if replace_values is None:
                    replace_values = (sp[i-1] + sp[i-2])/2 # find mean of the two adjacent spectra
                    bkg_1, params_1 = baseline_fitter.rubberband(replace_values)
                    spectra_corr = replace_values - bkg_1
                sp_corr[i] = replace_values
        sp_corr[i] = spectra_corr
        sp_base[i] = bkg_1

        if lim is not None and n >= lim:
            break
    
    if len(err_idx_lst)>0:
        print(f'Total data with errors : {len(err_idx_lst)}')

    if return_baselines:
        return sp_corr.T, err_idx_lst, sp_base.T
    else:
        return sp_corr.T, err_idx_lst