from torch.utils.data import DataLoader
import HistoricalPriceDataset as PriceData


debug_level = 3
pricedata = PriceData.HistPriceDataset(csvfile='./data/SP500.csv')

if debug_level >= 5:
    for row in range(len(pricedata)):
        record = pricedata[row]
        print(record)


dataloader = DataLoader(pricedata,
                        batch_size=1,
                        shuffle=False,
                        num_workers=4)


for i, x in enumerate(dataloader):
    print(x)
