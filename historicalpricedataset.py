import torch
from torch.utils.data.dataset import Dataset
import pandas as pd
import numpy as np


class HistPriceDataset(Dataset):
    """Historical Stock Price Dataset"""
    def __init__(self, csvfile, window=30, log=False):
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

        # Load the data
        self.price_history_frame = pd.read_csv(self.qual_filename)

        # Apply standard transformations
        self.roll_forward()
        self.calculate_differences()
        self.date_labels, self.prices, self.change = self.torch_it(False)

    def __len__(self):
        return len(self.price_history_frame) - self.window

    def __getitem__(self, idx):
        date_label = self.date_labels.iloc[idx+self.window]
        price = self.prices[idx:idx+self.window]
        change = self.change[idx:idx+self.window]

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
        sample = {'date_label': date_label, 'price': price, 'change': change}
        return sample

    def roll_forward(self):
        for i in range(1, len(self.price_history_frame)):
            if self.price_history_frame.iloc[i, 1] == 0.0:
                self.price_history_frame.iloc[i, 1] = self.price_history_frame.iloc[i-1, 1]
        print('Prices carried forward!')

    def calculate_differences(self):
        self.price_history_frame['change'] = np.nan
        for i in range(1, len(self.price_history_frame)):
            self.price_history_frame.iloc[i, 2] = np.float64((self.price_history_frame.iloc[i-1, 1] - self.price_history_frame.iloc[i, 1]) / self.price_history_frame.iloc[i-1, 1])
        print('Computed periodic price change!')

    def torch_it(self, convert):
        datelabels = self.price_history_frame.iloc[:, 0]
        if convert:
            prices = torch.from_numpy(np.array(self.price_history_frame.iloc[:, 1]))
            change = torch.from_numpy(np.array(self.price_history_frame.iloc[:, 2]))
        else:
            prices = self.price_history_frame.iloc[:, 1].as_matrix()
            change = self.price_history_frame.iloc[:, 2].as_matrix()

        return datelabels, prices, change
