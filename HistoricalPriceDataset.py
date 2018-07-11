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
        self.roll_forward()
        self.calculate_differences()
        self.date_labels, self.prices, self.change = self.torch_it()

    def __len__(self):
        return len(self.price_history)

    def __getitem__(self, idx):
        date_label = self.date_labels.iloc[idx, 0]
        price = self.prices.iloc[idx, 1]
        change = self.change.iloc[idx, 2]

        if self.log_transform:
            if self.log_transform == 10:
                price = np.log10(price)
                change = np.log10(change)
            elif self.log_transform == 2:
                price = np.log2(price)
                change = np.log2(change)
            elif self.log_transform == 'e':
                price = np.log(price)
                change = np.log(change)
            else:
                price = np.log10(price)
                change = np.log10(change)

        price_tensor = price
        change_tensor = change
        sample = {'date_label': date_label, 'price': price_tensor, 'change': change_tensor}
        return sample

    def roll_forward(self):
        for i in range(1, len(self.price_history)):
            if self.price_history.iloc[i, 1] == 0.0:
                self.price_history.iloc[i, 1] = self.price_history.iloc[i-1, 1]
        print('Prices carried forward!')

    def calculate_differences(self):
        self.price_history['change'] = None
        for i in range(1, len(self.price_history)):
            self.price_history.iloc[i, 2] = np.float64((self.price_history.iloc[i-1, 1] - self.price_history.iloc[i, 1]) / self.price_history.iloc[i-1, 1])
        print('Computed periodic price change!')

    def torch_it(self):
        datelabels = self.price_history.iloc[:, 0]
        prices = torch.from_numpy(np.array(self.price_history.iloc[:, 1]))
        print(self.price_history['change'])
        change = torch.from_numpy(np.array(self.price_history.iloc[:, 2]))
        return datelabels, prices, change
