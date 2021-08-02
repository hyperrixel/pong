from collections.abc import Iterable
from os import listdir
from os.path import isfile, join
import pickle
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
import torch


ALLOW_CUDA = True
BATCH_SIZE = 6
EPOCH_COUNT = 5000
TRAIN_DIR = 'data/img_train2'
LEARNING_RATE = 1e-6
LOSS_LIMIT = 30
LR_REDUCE_RATE = 0.5
ROOT_DIR = '.'

dim = 224
device = 'cuda' if ALLOW_CUDA and torch.cuda.is_available() else 'cpu'
print('Training on {}'.format(device))


class PongAutoEncoder1(torch.nn.Module):
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

        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.cnn1 = torch.nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=True)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.cnn2 = torch.nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=True)
        self.bn2 = torch.nn.BatchNorm2d(128)
        self.cnn3 = torch.nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=True)
        self.bn3 = torch.nn.BatchNorm2d(256)
        self.cnn4 = torch.nn.Conv2d(256, 512, 3, stride=1, padding=1, bias=True)
        self.bn4 = torch.nn.BatchNorm2d(512)
        self.cnn5 = torch.nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=True)
        self.bn5 = torch.nn.BatchNorm2d(512)
        self.cnn6 = torch.nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=True)
        self.bn6 = torch.nn.BatchNorm2d(512)
        self.cnn7 = torch.nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=True)
        self.bn7 = torch.nn.BatchNorm2d(512)
        self.fc1 = torch.nn.Linear(512, 2048, bias=True)
        self.tcnn1 = torch.nn.ConvTranspose2d(512, 512, 3, stride=1, padding=1,
                                              bias=True)
        self.dbn1 = torch.nn.BatchNorm2d(512)
        self.tcnn2 = torch.nn.ConvTranspose2d(512, 512, 3, stride=1, padding=1,
                                              bias=True)
        self.dbn2 = torch.nn.BatchNorm2d(512)
        self.tcnn3 = torch.nn.ConvTranspose2d(512, 512, 3, stride=1, padding=1,
                                              bias=True)
        self.dbn3 = torch.nn.BatchNorm2d(512)
        self.tcnn4 = torch.nn.ConvTranspose2d(512, 512, 3, stride=1, padding=1,
                                              bias=True)
        self.dbn4 = torch.nn.BatchNorm2d(512)
        self.tcnn5 = torch.nn.ConvTranspose2d(512, 256, 3, stride=1, padding=1,
                                              bias=True)
        self.dbn5 = torch.nn.BatchNorm2d(256)
        self.tcnn6 = torch.nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1,
                                              bias=True)
        self.dbn6 = torch.nn.BatchNorm2d(128)
        self.tcnn7 = torch.nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1,
                                              bias=True)
        self.dbn7 = torch.nn.BatchNorm2d(64)
        self.tcnn8 = torch.nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1,
                                              bias=True)
        self.dbn8 = torch.nn.BatchNorm2d(3)
        self.tcnn9 = torch.nn.ConvTranspose2d(3, 3, 3, stride=1, padding=1,
                                              bias=True)
        self.dbn9 = torch.nn.BatchNorm2d(3)


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

        x = torch.nn.functional.leaky_relu(self.maxpool(self.bn1(self.cnn1(x))),
                                           inplace=True)
        x = torch.nn.functional.leaky_relu(self.maxpool(self.bn2(self.cnn2(x))),
                                           inplace=True)
        x = torch.nn.functional.leaky_relu(self.maxpool(self.bn3(self.cnn3(x))),
                                           inplace=True)
        x = torch.nn.functional.leaky_relu(self.maxpool(self.bn4(self.cnn4(x))),
                                           inplace=True)
        x = torch.nn.functional.leaky_relu(self.maxpool(self.bn5(self.cnn5(x))),
                                           inplace=True)
        x = torch.nn.functional.leaky_relu(self.maxpool(self.bn6(self.cnn6(x))),
                                           inplace=True)
        x = torch.nn.functional.leaky_relu(self.maxpool(self.bn7(self.cnn7(x))),
                                           inplace=True)
        x = torch.flatten(x, 1)
        x = torch.tanh(self.fc1(x))
        x = x.view(-1, 512, 2, 2)
        x = torch.nn.functional.leaky_relu(self.dbn1(self.tcnn1(x)),
                                           inplace=True)
        x = torch.nn.functional.leaky_relu(
            torch.nn.functional.interpolate(self.dbn2(self.tcnn2(x)),
            scale_factor=1.5, recompute_scale_factor=True), inplace=True)
        x = torch.nn.functional.leaky_relu(
            torch.nn.functional.interpolate(self.dbn3(self.tcnn3(x)),
            scale_factor=2.5, recompute_scale_factor=True), inplace=True)
        x = torch.nn.functional.leaky_relu(
            torch.nn.functional.interpolate(self.dbn4(self.tcnn4(x)),
            scale_factor=2), inplace=True)
        x = torch.nn.functional.leaky_relu(
            torch.nn.functional.interpolate(self.dbn5(self.tcnn5(x)),
            scale_factor=2), inplace=True)
        x = torch.nn.functional.leaky_relu(
            torch.nn.functional.interpolate(self.dbn6(self.tcnn6(x)),
            scale_factor=2), inplace=True)
        x = torch.nn.functional.leaky_relu(
            torch.nn.functional.interpolate(self.dbn7(self.tcnn7(x)),
            scale_factor=2), inplace=True)
        x = torch.nn.functional.leaky_relu(
            torch.nn.functional.interpolate(self.dbn8(self.tcnn8(x)),
            scale_factor=2),inplace=True)
        x = torch.sigmoid(self.dbn9(self.tcnn9(x)))
        return x


class PongAutoEncoder2(torch.nn.Module):
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

        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.cnn1 = torch.nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=True)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.cnn2 = torch.nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=True)
        self.bn2 = torch.nn.BatchNorm2d(128)
        self.cnn3 = torch.nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=True)
        self.bn3 = torch.nn.BatchNorm2d(256)
        self.cnn4 = torch.nn.Conv2d(256, 512, 3, stride=1, padding=1, bias=True)
        self.bn4 = torch.nn.BatchNorm2d(512)
        self.cnn5 = torch.nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=True)
        self.bn5 = torch.nn.BatchNorm2d(512)
        self.cnn6 = torch.nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=True)
        self.bn6 = torch.nn.BatchNorm2d(512)
        self.cnn7 = torch.nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=True)
        self.bn7 = torch.nn.BatchNorm2d(512)
        self.fc1 = torch.nn.Linear(512, 2048, bias=True)
        self.tcnn1 = torch.nn.ConvTranspose2d(512, 512, 3, stride=2, padding=1,
                                              bias=True)
        self.dbn1 = torch.nn.BatchNorm2d(512)
        self.tcnn2 = torch.nn.ConvTranspose2d(512, 512, 3, stride=2, padding=0,
                                              bias=True)
        self.dbn2 = torch.nn.BatchNorm2d(512)
        self.tcnn3 = torch.nn.ConvTranspose2d(512, 512, 3, stride=2, padding=1,
                                              bias=True)
        self.dbn3 = torch.nn.BatchNorm2d(512)
        self.tcnn4 = torch.nn.ConvTranspose2d(512, 512, 3, stride=1, padding=0,
                                              bias=True)
        self.dbn4 = torch.nn.BatchNorm2d(512)
        self.tcnn5 = torch.nn.ConvTranspose2d(512, 256, 3, stride=1, padding=1,
                                              bias=True)
        self.dbn5 = torch.nn.BatchNorm2d(256)
        self.tcnn6 = torch.nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1,
                                              bias=True)
        self.dbn6 = torch.nn.BatchNorm2d(128)
        self.tcnn7 = torch.nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1,
                                              bias=True)
        self.dbn7 = torch.nn.BatchNorm2d(64)
        self.tcnn8 = torch.nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1,
                                              bias=True)
        self.dbn8 = torch.nn.BatchNorm2d(3)
        self.tcnn9 = torch.nn.ConvTranspose2d(3, 3, 3, stride=1, padding=1,
                                              bias=True)
        self.dbn9 = torch.nn.BatchNorm2d(3)


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

        x = torch.nn.functional.leaky_relu(self.maxpool(self.bn1(self.cnn1(x))),
                                           inplace=True)
        x = torch.nn.functional.leaky_relu(self.maxpool(self.bn2(self.cnn2(x))),
                                           inplace=True)
        x = torch.nn.functional.leaky_relu(self.maxpool(self.bn3(self.cnn3(x))),
                                           inplace=True)
        x = torch.nn.functional.leaky_relu(self.maxpool(self.bn4(self.cnn4(x))),
                                           inplace=True)
        x = torch.nn.functional.leaky_relu(self.maxpool(self.bn5(self.cnn5(x))),
                                           inplace=True)
        x = torch.nn.functional.leaky_relu(self.maxpool(self.bn6(self.cnn6(x))),
                                           inplace=True)
        x = torch.nn.functional.leaky_relu(self.maxpool(self.bn7(self.cnn7(x))),
                                           inplace=True)
        x = torch.flatten(x, 1)
        x = torch.tanh(self.fc1(x))
        x = x.view(-1, 512, 2, 2)
        x = torch.nn.functional.leaky_relu(self.dbn1(self.tcnn1(x)),
                                           inplace=True)
        x = torch.nn.functional.leaky_relu(torch.nn.functional.interpolate(
            self.dbn2(self.tcnn2(x)), scale_factor=2), inplace=True)
        x = torch.nn.functional.leaky_relu(torch.nn.functional.interpolate(
            self.dbn3(self.tcnn3(x)), scale_factor=2), inplace=True)
        x = torch.nn.functional.leaky_relu(torch.nn.functional.interpolate(
            self.dbn4(self.tcnn4(x)), scale_factor=2), inplace=True)
        x = torch.nn.functional.leaky_relu(torch.nn.functional.interpolate(
            self.dbn5(self.tcnn5(x)), scale_factor=2), inplace=True)
        x = torch.nn.functional.leaky_relu(self.dbn6(self.tcnn6(x)),
                                           inplace=True)
        x = torch.nn.functional.leaky_relu(self.dbn7(self.tcnn7(x)),
                                           inplace=True)
        x = torch.nn.functional.leaky_relu(self.dbn8(self.tcnn8(x)),
                                           inplace=True)
        x = torch.sigmoid(self.dbn9(self.tcnn9(x)))
        return x


def list_images() -> list:
    """
    Create list of image files
    ==========================

    Returns
    -------
    list
        List of existing files.
    """

    return ['{}/{}'.format(TRAIN_DIR, f) for f in listdir(TRAIN_DIR)
                                         if isfile('{}/{}'.format(TRAIN_DIR,
                                                                  f))]


def load_image_np(file_name : str) -> np.ndarray:
    """
    Load image as numpy.ndarray

    Parameters
    ----------
    file_name : str
        File path to load.

    Returns
    -------
    np.ndarray
        The loaded image as numpy.ndarray.
    """

    with open(file_name, 'rb') as instream:
        return np.expand_dims(pickle.load(instream), axis=0)


def train_model():
    """
    Perform model training
    ======================
    """

    imagelist = list_images()
    model = PongAutoEncoder2()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.L1Loss()
    best_loss = np.inf
    bad_count = 0
    epoch_loss = 0.0
    for epoch in range(1, EPOCH_COUNT + 1):
        shuffle(imagelist)
        batch_count = 0
        loss_items = []
        for batch in yield_x_batches(imagelist, BATCH_SIZE):
            batch_size = len(batch)
            np_images = np.asarray(np.concatenate([load_image_np(b)
                                                  for b in batch]),
                                                  dtype=np.float32) / 255
            x = torch.from_numpy(np_images).to(device)
            y = torch.from_numpy(np_images).to(device)
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
                       join(ROOT_DIR,
                            'results/PongAutoEncoder_best.state_dict'))
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
        with open('pong_train.log', 'a', encoding='utf8') as outstream:
            outstream.write('{}\n'.format(print_str))
        f = plt.figure()
        imgs = y_hat.detach().cpu().numpy()
        imgs *= 255
        imgs = np.moveaxis(np.asarray(imgs, dtype=np.uint8), 1, 3)
        imgs_length = imgs.shape[0]
        if imgs_length < 5:
            width = imgs_length
        else:
            width = 5
        for i in range(width):
            p = f.add_subplot(1, width, i+1)
            plt.imshow(imgs[-i])
            plt.axis('off')
        plt.savefig('results/fig/{:05d}.png'.format(epoch), bbox_inches='tight',
                    pad_inches=0.1, dpi=400)


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
