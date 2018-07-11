import torch
from torch.utils.data.dataset import Dataset
import pandas as pd
import numpy as np
import os

class HistPriceDataset(Dataset):
    """Historical Stock Price Dataset"""
    def __init__(self, csvfile, rootdir=None, log=False):
        """
        :param csvfile:
        :param rootdir:
        :param log:
        """
        super().__init__()
        self.qual_filename = csvfile
        if rootdir:
            self.qual_filename = os.path.join(rootdir, self.qual_filename)
        self.price_history = pd.read_csv(self.qual_filename)
        self.root_dir = rootdir
        self.log_transform = log

    def __len__(self):
        return len(self.price_history)

    def __getitem__(self, idx):
        date = self.price_history.iloc[idx, 1].as_matrix()
        price = self.price_history.iloc[idx, 2].as_matrix()

        if self.log_transform:
            if self.log_transform == 10:
                price = np.log10(price)
            elif self.log_transform == 2:
                price = np.log2(price)
            elif self.log_transform == 'e':
                price = np.log(price)
            else:
                price = np.log10(price)

        sample = {'date': date, 'price':price}
        return sample

