import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import pandas as pd
import random

class HistPriceDataset(Dataset):
    """Historical Stock Price Dataset"""
    def __init__(self, csvfile, window=30, log=False, valid_pct=0.2):
        """
        :param csvfile:
        :param window:
        :param log:
        """
        super().__init__()
        # Store the input parameters
        self.window = window
        self.qual_filename = csvfile
        self.log_transform = log

        rawfram = pd.read_csv(self.qual_filename, header=0, thousands=r',')

        # Load the data
        self.rawprices = rawfram['Price'].as_matrix()

        self.open = rawfram['Open'].as_matrix()

        self.high = rawfram['High'].as_matrix()

        self.low = rawfram['Low'].as_matrix()

        self.date_strings = rawfram['Date']

        x = int((1.0-valid_pct)*len(self.rawprices))
        self.is_train_set = x

        # Apply standard transformations
        self.roll_forward()

        # self.rawprices = np.concatenate((self.rawprices, self.is_train_set), axis=2)

    def __len__(self):
        return len(self.rawprices) - self.window

    def __getitem__(self, idx):
        ix = int(idx)+self.window
        validation_ix_start = self.is_train_set
        print(self.rawprices.shape)
        price = self.rawprices[idx:ix]
        if idx<=validation_ix_start:
            train_price = self.rawprices[idx:ix]
        else:
            validation_price = self.rawprices[idx:ix]

        if self.log_transform:
            if self.log_transform == 10:
                price = np.log10(price)
            elif self.log_transform == 2:
                price = np.log2(price)
            elif self.log_transform == 'e':
                price = np.log(price)
            else:
                price = np.log10(price)
        return train_price, validation_price

    def roll_forward(self):
        for i in range(1, len(self.rawprices)):
            if self.rawprices[i] == 0.0:
                self.rawprices[i] = self.rawprices[i-1]
            if self.open[i] == 0.0:
                self.open[i] = self.open[i-1]
            if self.high[i] == 0.0:
                self.high[i] = self.high[i-1]
            if self.low[i] == 0.0:
                self.low[i] = self.low[i-1]
        print('Prices carried forward!')

