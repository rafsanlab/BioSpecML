"""
This part is from a common_functions library available:
https://github.com/rafsanlab/etc/blob/main/common_functions.py

"""
import pathlib
import pandas as pd

def get_data(path:str, file_type:str, verbose:bool=True):
    """
    Get files from path with spesific file types.

    Args:
    - path (str): string path containing the files.
    - file_type (str): file type i.e; '.png'.
    - verbose (bool): condition of output summary.

    Return:
    - paths (list): list of paths.

    """

    path = pathlib.Path(path)
    lookfor = f'*{file_type}'
    paths = list(path.glob(lookfor))
    paths = sorted(paths)
    if verbose == True:
        print(f'Total paths: {len(paths)}')

    return paths


def compile_data_csv(path_list:list, read_delimiter:str='/t', save_delimiter:str='/t',
                     col_to_int:bool=False, save_data:bool=False, savedir:str=None,
                     ):
    
    """
    Read paths of csv files from a list and compiled into one df.

    """
    main_df = pd.DataFrame()
    for path in path_list:
        df_ = pd.read_csv(path, delimiter=read_delimiter)
        main_df = pd.concat([main_df, df_], axis=0)
        main_df = main_df.reset_index(drop=True)

    if col_to_int:
        numeric_cols = [int(round(float(c))) if str(c).replace('.','').isdigit() else c for c in main_df.columns]
        main_df.columns = numeric_cols

    if save_data:
        if savedir==None:
            savedir = os.path.join(os.getcwd(), 'main_df.tsv')
        main_df.to_csv(savedir, sep=save_delimiter)
    
    return main_df