from torchvision import transforms
from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

class TabularMILDataset(Dataset):
    def __init__(self, df, label_col:str, bags_id_col:str, label_dict:dict=None,
                 metadata_cols:list=None, 
                 transform_data:bool=True, mean=None, std=None,
                 unsqueeze:bool=False):
        """
        A simple MIL dataset for instances in a bag learning.
        Must supply label_col and bags_id_col that is unique.
        Option to convert bag label using label_dict.

        """
        self.df = df
        self.label_col = label_col
        self.bags_id_col = bags_id_col
        self.label_dict = label_dict
        self.bags_ids = df[bags_id_col].unique()
        self.metadata_cols = metadata_cols
        self.transform_data = transform_data
        self.mean = mean
        self.std = std
        self.unsqueeze = unsqueeze

    def __len__(self):
        return len(self.bags_ids)

    def __getitem__(self, idx):

        # get item from bags id and the data
        bag_id = self.bags_ids[idx]
        bag_data = self.df[self.df[self.bags_id_col]==bag_id]
        
        # get bag labels
        if self.label_dict != None:
            bag_labels = [l for l in bag_data[self.label_col].map(self.label_dict).unique()]
        else:
            bag_labels = [l for l in bag_data[self.label_col].unique()]

        # prepare bag labels
        bag_labels = [torch.tensor(l, dtype=torch.float32) for l in bag_labels]
        bag_labels = bag_labels[0]

        # drop metadata from data
        if self.metadata_cols != None:
            self.metadata_cols.extend(self.label_col)
            self.metadata_cols.extend(self.bags_id_col)
        else:
            self.metadata_cols = [self.label_col, self.bags_id_col]
        metadata_cols_ = [c for c in self.metadata_cols if c in bag_data.columns]
        bag_data = bag_data.drop(metadata_cols_, axis=1).values
        
        # get finalise data without metadata
        bag_data = torch.tensor(bag_data, dtype=torch.float32)

        # option to transform the data
        if self.transform_data:
            bag_data = (bag_data - self.mean) / self.std
            bag_data = bag_data.to(torch.float32)

        return bag_data, bag_labels