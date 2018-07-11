import os
from torch.utils.data import DataLoader
import HistoricalPriceDataset as PriceData

pricedata = PriceData.HistPriceDataset(csvfile='./data/SP500.csv')

for row in range(len(pricedata)):
    record = pricedata[row]
    print(record)


