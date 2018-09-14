import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import historicalpricedataset as prices
import numpy as np

debug_level = 3
window_size = 15  # The window size includes the target value at position -1
hidden_size = [6, 3]
batch_size = 100
EPOCHS = 200
epoch_loss = 0.0
print_at_batch_intervals = False


class LstmModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=window_size-1, hidden_size=hidden_size[0])
        self.lstm2 = nn.LSTM(input_size=hidden_size[0], hidden_size=hidden_size[1])
        self.linear = nn.Linear(hidden_size[1], 1)

    def forward(self, forward_input, future=0):
        outputs = []

        h_t1 = torch.randn(forward_input.size(1), forward_input.size(1),
                           hidden_size[0], dtype=torch.double)
        c_t1 = torch.zeros(forward_input.size(1), forward_input.size(1),
                           hidden_size[0], dtype=torch.double)
        h_t2 = torch.randn(h_t1.size(0), forward_input.size(1),
                           hidden_size[1], dtype=torch.double)
        c_t2 = torch.zeros(c_t1.size(0), forward_input.size(1),
                           hidden_size[1], dtype=torch.double)

        for ix, input_t in enumerate(forward_input.chunk(forward_input.size(1), dim=2)):
            # if ix == 0:
            #     print("Input is of size %s" % str(input_t.size()))
            #     print(input[:, 0, :])
            # print(input_t.size(0), input_t.size(1), input_t.size(2))
            # h_t1, c_t1 = self.lstm1(input_t)
            h_t1, c_t1 = self.lstm1(input_t, (h_t1, c_t1))
            # print(c_t1)
            # print("Layer 1: %i, %i" % (len(h_t1), len(c_t1)))

            # h_t2, c_t2 = self.lstm2(h_t1)
            h_t2, c_t2 = self.lstm2(h_t1, (h_t2, c_t2))
            # print("Layer 2: %i, %i" % (len(h_t2), len(c_t2)))
            # print(h_t2.size())
            output = self.linear(h_t2)
            # if ix == 0:
            #     print("output is of type: %s and of size %s" % (type(output), output.size()))
            #     print(output[:, 0, 0])
            outputs.extend([output])
        # print("outputs is of type: %s" % type(outputs))
        # print("Elements in outputs are of type: %s and of size %s" % (outputs[0], outputs[0].size()))
        outputs = torch.stack(outputs, 1)
        # print("Pre-squeeze outputs is of size: %s" % str(outputs.size()))
        outputs = outputs.squeeze(3)
        # print()
        # print("Returned outputs is of type: %s" % type(outputs))
        # print("Returned outputs is of size: %s" % str(outputs.size()))
        # exit()
        return outputs


if __name__ == '__main__':
    # set random seeds to 0
    np.random.seed(0)
    torch.manual_seed(0)

    # Create the Dataset Object
    pricedata = prices.HistPriceDataset(csvfile='./data/S&P 500 Historical Data.csv',
                                        window=window_size,
                                        log=False)

    if debug_level >= 6:
        for row in range(len(pricedata)):
            record = pricedata[row]
            print(record)

    # Create the data loader
    dataloader = DataLoader(pricedata, batch_size=batch_size, shuffle=True, num_workers=4)

    if debug_level >= 5:
        for i, d in enumerate(dataloader):
            if i == 0:
                print(len(d))
            print(i, d)

    # build the model
    model = LstmModel().double()

    criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    # optimizer = optim.LBFGS(model.parameters(), lr=0.001)
    # optimizer = optim.Adam(model.parameters(), weight_decay=0.1, lr=0.01)
    optimizer = optim.SGD(model.parameters(), weight_decay=0.1, lr=0.005)


    # Begin training
    for i in range(EPOCHS):
        epoch_loss = 0.0
        print('EPOCH: ', i)
        # max_j = len(dataloader)-1
        for j, open_batch in enumerate(dataloader):
            print(open_batch.shape)
            # print("J: %i" % max_j)
            # print("Batch is of type: %s and has a shape of %s" % (type(batch), batch.shape))
            input = open_batch[:, 0:-1].unsqueeze_(1)
            # print("Input is of type: %s and has a shape of %s" % (type(input), input.shape))
            target = open_batch[:, -1:].unsqueeze_(1)
            # print("Target is of type: %s and has a shape of %s" % (type(target), target.shape))

            optimizer.zero_grad()
            out = model(input)
            # print("Out size: %s" % str(out.size()))
            # print(out)
            # print("Target size: %s" % str(target.size()))
            loss = criterion(out, target)
            # print('Epoch: %i, Batch ID: %i,  loss: %0.7f, epoch loss: %0.3f' % (i, j, loss.item(), epoch_loss))
            if (j + 1) % 10 == 0 or j == max_j:
                if print_at_batch_intervals:
                    print('Epoch: %i, Batch ID: %i,  loss: %0.7f' % (i, j, loss.item()))
                # print('Epoch: %i, Batch ID: %i,  loss: %0.7f' % (i, j, np.log10(loss.item())))
            epoch_loss += float(loss.item())
            loss.backward()

            optimizer.step()
        print("Loss for epoch = %0.4f" % epoch_loss)
