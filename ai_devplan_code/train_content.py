from collections.abc import Iterable
import pickle

import numpy as np
import torch


ALLOW_CUDA = True
BATCH_SIZE = 128
EPOCH_COUNT = 5000
TRAIN_DIR = 'data/img_train2'
LEARNING_RATE = 1e-4
LOSS_LIMIT = 10
LR_REDUCE_RATE = 0.5
ROOT_DIR = '.'

dim = 256
device = 'cuda' if ALLOW_CUDA and torch.cuda.is_available() else 'cpu'
print('Training on {}'.format(device))


class PongContentEncoder(torch.nn.Module):
    """
    Pong encoder-decoder network
    ============================
    """

    def __init__(self):
        """
        Initialize an instance of the object
        ====================================
        """

        super().__init__()
        self.e_fc1 = torch.nn.Linear(2081, 1024)
        self.e_bn1 = torch.nn.BatchNorm1d(1024)
        self.e_fc2 = torch.nn.Linear(1024, 512)
        self.e_bn2 = torch.nn.BatchNorm1d(512)
        self.e_fc3 = torch.nn.Linear(512, 512)
        self.e_bn3 = torch.nn.BatchNorm1d(512)
        self.e_fc4 = torch.nn.Linear(512, 384)
        self.e_bn4 = torch.nn.BatchNorm1d(384)
        self.m_fc = torch.nn.Linear(384, 384)
        self.d_fc1 = torch.nn.Linear(384, 512)
        self.d_bn1 = torch.nn.BatchNorm1d(512)
        self.d_fc2 = torch.nn.Linear(512, 512)
        self.d_bn2 = torch.nn.BatchNorm1d(512)
        self.d_fc3 = torch.nn.Linear(512, 1024)
        self.d_bn3 = torch.nn.BatchNorm1d(1024)
        self.d_fc4 = torch.nn.Linear(1024, 2081)
        self.dropout = torch.nn.Dropout(p=0.2)


    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass
        ====================

        Parameters
        ----------
        x : torch.Tensor
            Input data to be forwarded.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        x = torch.nn.functional.leaky_relu(self.e_bn1(self.e_fc1(x)), inplace=True)
        x = self.dropout(torch.nn.functional.leaky_relu(self.e_bn2(self.e_fc2(x)), inplace=True))
        x = self.dropout(torch.nn.functional.leaky_relu(self.e_bn3(self.e_fc3(x)), inplace=True))
        x = self.dropout(torch.nn.functional.leaky_relu(self.e_bn4(self.e_fc4(x)), inplace=True))
        x = torch.tanh(self.m_fc(x))
        x = self.dropout(torch.nn.functional.leaky_relu(self.d_bn1(self.d_fc1(x)), inplace=True))
        x = self.dropout(torch.nn.functional.leaky_relu(self.d_bn2(self.d_fc2(x)), inplace=True))
        x = self.dropout(torch.nn.functional.leaky_relu(self.d_bn3(self.d_fc3(x)), inplace=True))
        x = torch.sigmoid(self.d_fc4(x))

        return x


def train_model():
    """
    Perform model training
    ======================
    """

    with open('content_dataset.pkl', 'rb') as instream:
        records = pickle.load(instream)
    model = PongContentEncoder()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.L1Loss()
    best_loss = np.inf
    bad_count = 0
    epoch_loss = 0.0
    for epoch in range(1, EPOCH_COUNT + 1):
        batch_count = 0
        loss_items = []
        for batch in yield_x_batches(records, BATCH_SIZE):
            batch_size = len(batch)
            batch = np.asarray(batch, dtype=np.float32)
            x = torch.from_numpy(batch).to(device)
            y = torch.from_numpy(batch).to(device)
            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            batch_count += 1
            batch_loss = float(loss.item())
            for i in range(batch_size):
                loss_items.append(batch_loss)
            print_str = '{}/{} bl: {:.8f} el: {:.8f} lr: {:e}'.format(epoch,
                        batch_count, batch_loss, epoch_loss,
                        optimizer.param_groups[0]['lr'])
            print('\r{}                                 '
                  .format(print_str), end='')
        epoch_loss = sum(loss_items) / len(loss_items)
        print_str = '{}/{} bl: {:.8f} el: {:.8f} lr: {:e}'.format(epoch,
                    batch_count, batch_loss, epoch_loss,
                    optimizer.param_groups[0]['lr'])
        if epoch_loss < best_loss:
            torch.save(model.state_dict(),
                            'results/PongContentEncoder_best.state_dict')
            best_loss = epoch_loss
            print_str += ' - best model saved'
            print('\r{}                                 '
                  .format(print_str), end='')
            bad_count = 0
        else:
            print('\r{}                                 '
                  .format(print_str), end='')
            bad_count += 1
        if bad_count == LOSS_LIMIT:
            optimizer.param_groups[0]['lr'] *= LR_REDUCE_RATE
        with open('pong_train_content.log', 'a', encoding='utf8') as outstream:
            outstream.write('{}\n'.format(print_str))


def yield_x_batches(x : Iterable, batch_size : int,
                    drop_last : bool =False) -> Iterable:
    """
    Yield batches of data
    =====================

    Parameters
    ----------
    x : Iterable
        Inputs data.
    batch_size : int
        Number of data elements in a batch.
    drop_last : bool, optional (False if omitted)
        Whether to drop or not the last element of a batch.

    Yields
    ------
    Iterable
        Slice of x elements.
    """

    pos = 0
    total_len = len(x)
    while pos + batch_size < total_len:
        yield x[pos:pos + batch_size]
        pos += batch_size
    if pos < total_len and not drop_last:
        yield x[pos:]


train_model()
