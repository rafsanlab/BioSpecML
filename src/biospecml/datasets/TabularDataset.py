from torchvision import transforms
from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

class TabularDataset(Dataset):
    def __init__(self, df=None, file_path:str=None, file_delimiter:str='\t',
                 index_col=None, label_col:str='', metadata_cols:list=None,
                 label_dict:dict=None,  transform_data:bool=True,
                 mean=None, std=None, unsqueeze:bool=False):
        """
        Read or accept df tabular data, must have label column for each rows.
        Options to convert labels to value of a dictionary by supplying label_dict.
        Mean and SD data transformation is supported.
        And options to squeeze the data to fit into Conv1D network.

        """
        # get df or read dataframe
        if df != None:
            self.df = df
        elif file_path != None:
            self.df = pd.read_csv(file_path, delimiter=file_delimiter, index_col=index_col)
        else:
            raise Exception('Please provide *df or *file_path argument.')

        # get labels and option to map labels to label_dict
        if label_dict != None and label_col != '':
            self.labels = self.df[label_col].map(label_dict)
        else:
            self.labels = self.df[label_col]
        self.labels = self.labels.values.astype(np.float32)

        # drop metadata, if None add label_col to metadata
        if metadata_cols != None:
            metadata_cols.extend(label_col)
        else:
            metadata_cols = [label_col]
        metadata_cols_ = [col for col in metadata_cols if col in self.df.columns]
        if metadata_cols_!=None:
            self.df = self.df.drop(metadata_cols_, axis=1)
        self.df = self.df.values.astype(np.float32)

        # other variables
        self.transform_data = transform_data
        self.mean, self.std = mean, std
        self.unsqueeze = unsqueeze

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        # get data and label
        data = torch.tensor(self.df[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        # option to transform the data
        if self.transform_data:
            data = (data - self.mean) / self.std
            data = data.to(torch.float32)
        
        # option to unsquuze the data for 1D convolution
        if self.unsqueeze:
            data = data.unsqueeze(0)

        return data, label