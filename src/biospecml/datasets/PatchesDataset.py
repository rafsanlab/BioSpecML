import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from torchvision import transforms

class PatchesDataset(Dataset):
    def __init__(self,
                 folder_path,
                 meta_file_path,
                 target_col='RFS_event',
                 fname_col='Fname',
                 ftype=".npz",
                 std_mean_npz_file=None,
                 transpose_channel:bool=True,
                 ):
        self.folder_path = folder_path
        self.ftype = ftype
        self.target_col = target_col
        self.fname_col = fname_col
        self.std_mean_npz_file = std_mean_npz_file
        self.transpose_channel = transpose_channel

        # list all non-hidden files with the correct extension
        self.file_list = [f for f in os.listdir(folder_path)
                          if f.endswith(self.ftype) and not f.startswith('.')]

        # read metadata from the file that contain 'Fname' and label
        self.df_meta = pd.read_csv(meta_file_path, delimiter='\t', index_col=0)

        # get the mean and std, data must be (i, 3, x, y)
        if self.std_mean_npz_file is not None:
            data = np.load(self.std_mean_npz_file)
            self.mean = np.array(data['mean']).reshape(1, -1, 1, 1)
            self.std = np.array(data['std']).reshape(1, -1, 1, 1)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):

        # open the data and normalise it
        fpath = os.path.join(self.folder_path, self.file_list[idx])
        data = np.load(fpath)
        patches = torch.tensor(data['patches'], dtype=torch.float32)
        if self.transpose_channel:
            patches =  np.transpose(patches, (0, 3, 1, 2))
        if self.std_mean_npz_file is not None:
            patches = (patches - torch.tensor(self.mean, dtype=torch.float32)) / torch.tensor(self.std, dtype=torch.float32)

        # get label
        fname = os.path.basename(fpath).split('.')[0]
        label = self.df_meta.loc[self.df_meta[self.fname_col] == fname, self.target_col].values[0]
        label = torch.tensor(label, dtype=torch.int64)

        return patches, label
    