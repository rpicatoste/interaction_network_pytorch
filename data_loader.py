import numpy as np
import torch

from physics_engine import VEL


class DataLoader:

    def __init__(self, data, batch_size, shuffle=True, device='cuda'):

        # if not isinstance(data, list):
        #     data = [data]

        # self.data = [torch.from_numpy(sample).type(torch.float32).to(device) for sample in data]
        self.data = torch.from_numpy(data).type(torch.float32).to(device)

        self.batch_size = batch_size

        self.n_samples = len(data)
        self.n_batches = int(np.ceil(self.n_samples / self.batch_size))

        self.n_states = data.shape[1]
        self.n_objects = data.shape[2]

        self.shuffle = shuffle
        self.indexes = np.arange(self.n_samples-1)
        self.apply_shuffle()

    def __len__(self):
        return self.n_batches

    def __iter__(self):

        def iterator():
            self.apply_shuffle()
            for index in range(self.n_batches):
                X, y = self[index]

                yield X, y

        return iterator()

    def __getitem__(self, index):
        if index >= self.n_batches:
            raise IndexError(f'Index {index} exceeds highest index {self.n_batches-1}.')

        start = index * self.batch_size
        end = min(start + self.batch_size, self.n_samples - 1)

        indexes_X = self.indexes[start:end]
        indexes_y = np.array([idx+1 for idx in indexes_X])

        X = self.data[indexes_X]
        y = self.data[indexes_y]
        y = y[:, VEL]

        return X, y

    def apply_shuffle(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
