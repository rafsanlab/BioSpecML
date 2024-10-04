from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm

def paths_splitter(df_paths, path_col, label_col, test_size:float=0.2,
                   split_method:str='random', stratify_col=None, random_state:int=42,
                   manual_train_list:list|None=None, manual_test_list:list|None=None,
                   verbose:bool=True,
                    ):
    """
    Args:
    - df_paths (pd.DataFrame): containing columns: [path_col|label_col]
    - path_col (str): column name containing file paths
    - label_col (str): column name containing labels or classes
    - test_size (float, optional):  test split proportion, default is 0.2.
    - split_method (str, optional): method used to split the data. Options are 'random' for 'manual'.
    - stratify_col (str, optional): column name to stratify by during splitting. Default is None.
    - random_state (int, optional): seed used by the random number generator. Default is 42.
    - manual_train_list (list, optional): list of file paths to include in the training set. If provided, overrides the test_size and split_method parameters. Default is None.
    - manual_test_list (list, optional): list of file paths to include in the test set. If provided, overrides the test_size and split_method parameters. Default is None.
    - verbose (bool, optional): whether to print information about the split. Default is True.

    Returns:
    - train_paths (pd.DataFrame): DataFrame containing training set paths and labels.
    - test_paths (pd.DataFrame): DataFrame containing test set paths and labels.

    """
    if split_method == 'random':

        # stratify condition
        if stratify_col!=None:
            stratify_col = df_paths[stratify_col]
        
        # split the paths
        X_train, X_test, y_train, y_test = train_test_split(
            df_paths[path_col], df_paths[label_col],
            test_size=test_size, random_state=random_state,
            stratify=stratify_col
            )
        
        # create dfs from the splitted path
        train_paths = pd.DataFrame({path_col:X_train, label_col:y_train})
        test_paths = pd.DataFrame({path_col:X_test, label_col:y_test})

    elif split_method == 'manual':

        train_paths, test_paths = pd.DataFrame(), pd.DataFrame()

        for label, _ in df_paths[label_col].value_counts().items():

            # search label in the train list
            if label in manual_train_list:
                df_ = df_paths[df_paths[label_col]==label]
                train_paths = pd.concat([train_paths, df_])

            # search label in the test list
            elif label in manual_test_list:
                df_ = df_paths[df_paths[label_col]==label]
                test_paths = pd.concat([test_paths, df_])
    
    else:
        raise Exception('Input for *split_method is invalid.')

    # condition of versone
    if verbose:
        print('Value counts of train paths :')
        print(train_paths[label_col].value_counts(),'\n')
        print('Value counts of test paths :')
        print(test_paths[label_col].value_counts())

    return train_paths, test_paths


def calc_ds_mean_std(df, replace_zeros:bool=True, replace_nans:bool=True, 
                     replace_inf:bool=True, replace_value:float=1.4013e-45):
    """
    Calculate the mean and standard deviation of numeric columns in a DataFrame.
    [*] Expects features as columns, readings are rows.

    Args:
    - df (pd.DataFrame): Input DataFrame.
    - replace_zeros (bool): Replace zeros with `replace_value`. Default is True.
    - replace_nans (bool): Replace NaNs with `replace_value`. Default is True.
    - replace_value (float): Value used for replacement of zeros and NaNs.

    Returns:
    - mean_values, std_values (pd.Series, pd.Series):
        - mean and standard deviation values for each numeric column.

    """
    mean_values = df.mean(numeric_only=True)
    std_values = df.std(numeric_only=True)

    # check if NaNs, zeros, or infinities are present
    if mean_values.isna().any() or std_values.isna().any():
        print("calc_ds_mean_std(): NaNs detected in mean_values or std_values")
    if (mean_values.eq(0) & std_values.eq(0)).any():
        print("calc_ds_mean_std(): Zeros detected in mean_values or std_values")
    if (mean_values.isin([np.inf, -np.inf]) | std_values.isin([np.inf, -np.inf])).any():
        print("calc_ds_mean_std(): Infinities detected in mean_values or std_values")
    
    # replace it with replace_values
    if replace_zeros==True:
        mean_values.replace(0, replace_value, inplace=True)
        std_values.replace(0, replace_value, inplace=True)
    if replace_nans==True:
        mean_values.fillna(replace_value, inplace=True)
        std_values.fillna(replace_value, inplace=True)
    if replace_inf:
        mean_values.replace([np.inf, -np.inf], replace_value, inplace=True)
        std_values.replace([np.inf, -np.inf], replace_value, inplace=True)
    return mean_values, std_values


def upsampling_via_smote(X, y, bygroup:bool=False, y_col=None, metadata_cols:list=[], random_state:int=42, sampling_strategy:str='auto'):
    """
    SMOTE on X matrix.
    - X either numpy array or dataframe.
    - bygroup (bool) : allow to return X with metadata, X must be a dataframe
        and y_col must represent y. Set to True only when the metadata in X 
        correspond is as unique as y_col.

    """
    if isinstance(X, pd.DataFrame):
        # convert col to str to avoid being upsampled
        X.columns = X.columns.astype(str)
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    
    if bygroup==False:
        X_resampled, y_resampled = smote.fit_resample(X, y)
    
    elif bygroup==True:
        metadata_cols_ = [c for c in X.columns if c in metadata_cols]
        X_meta = X[metadata_cols_].drop_duplicates()
        X = X.drop(metadata_cols_, axis=1)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # expand metadata to the upsampled y
        y_resampled_meta = pd.DataFrame()
        for i in y_resampled.unique(): # iterate each label  
            x_meta = X_meta[X_meta[y_col]==i] # get a single row from metadata that has label y
            y_meta = y_resampled[y_resampled==i].to_frame() # filter the y_resampled that the label y
            x_meta = pd.concat([x_meta]*len(y_meta), ignore_index=True) # expands the single row to match filtered y_resampled
            x_meta.index = y_meta.index # make the expanded rows to have the same index
            if y_col in x_meta.columns: # remove label y column if it exist in the expanded rows
                x_meta.drop(y_col, axis=1, inplace=True)
            y_meta = pd.concat([y_meta, x_meta], axis=1) # combine the expanded rows
            y_resampled_meta = pd.concat([y_resampled_meta, y_meta], axis=0) # save meta labels to main df
        
        y_resampled = y_resampled_meta

    return X_resampled, y_resampled


def create_tabular_bags(
        df, label_col:str, num_bags:int, bags_id_connector:str='_',
        bags_id_col:str='BagsID', bags_idx_col='BagsIDX', verbose:bool=True,
        ):
    """
    Turn a tabular data into bags with instances based on required bags per label.
    Return bags_df with unique bags_id_col as bag identifier.

    """
    bags_df, bags_list = pd.DataFrame(), [] # variables
    grouped_df = df.groupby(label_col) # grouped df by label_col
    counter_idx = 0
    for label, group in grouped_df:
        counter = 0 # counter for bags id
        if verbose:
            print(f'Bag: {label},\t total instances: {group.shape}')
        bags = np.array_split(group, num_bags) # split group based on bags number
        for bag in bags:
            counter += 1
            counter_idx += 1
            bag[bags_id_col] = str(bag[label_col].unique()[0])+f'{bags_id_connector}{counter}' # unique bags id
            bag[bags_idx_col] = counter_idx
        bags_list.extend(bags) # compile all bags
    bags_df = pd.concat(bags_list) # turn all bags to single df
    return bags_df


def calculate_mean_std_imgs(image_folder, normalise_to_ones_zeros:bool=False):

    means, stds = [], []

    # Iterate through the image files in the folder
    for img_name in tqdm(os.listdir(image_folder)):

        if img_name.startswith('.'):  # Skip hidden files
            continue
        img_path = os.path.join(image_folder, img_name)

        # Open the image and convert it to NumPy array
        img = Image.open(img_path)
        if normalise_to_ones_zeros:
            img = np.array(img) / 255.0  # Normalize to [0, 1] range

        # Calculate the mean and std across the height and width dimensions (axis=(0,1))
        # This computes mean/std for each channel (RGB)
        mean = np.mean(img, axis=(0, 1))
        std = np.std(img, axis=(0, 1))

        means.append(mean)
        stds.append(std)

    # Compute overall mean and std by averaging over all images
    mean = np.mean(means, axis=0)
    std = np.mean(stds, axis=0)

    return mean, std