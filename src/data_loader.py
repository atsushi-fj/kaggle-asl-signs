import torch
import numpy as np
import gc
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os

from .utils import load_config


# Load config file 
cfg = load_config(file="config.yaml")

class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i].astype(np.float32), self.y[i]


def data_split(X, y,
               test_size=0.2,
               random_state=42,
               stratify=None):
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                    test_size=test_size,
                                                    random_state=random_state,
                                                    stratify=stratify)
    del X, y
    gc.collect()
    train_dataset = Dataset(X_train, y_train)
    val_dataset = Dataset(X_val, y_val)
    
    return train_dataset, val_dataset


def create_dataloader(train_dataset, val_dataset,
                      batch_size=128,
                      num_workers=os.cpu_count(),
                      pin_memory=True,
                      train_drop_last=True):
    
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  drop_last=train_drop_last)
    
    val_dataloader = DataLoader(val_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory)
    
    return train_dataloader, val_dataloader
    
    