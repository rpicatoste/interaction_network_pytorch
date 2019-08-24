import numpy as np
import torch

from physics_engine import VEL


class DataLoader:

    def __init__(self, data, batch_size, shuffle=True, device='cuda'):

        if not isinstance(data, list):
            data = [data]

        self.data = [torch.from_numpy(sample).type(torch.float32).to(device) for sample in data]
        self.n_samples = sum([len(sample)-1 for sample in self.data])
        self.batch_size = batch_size if batch_size is not None else self.n_samples
        self.n_batches = int(np.ceil(self.n_samples / self.batch_size))
        self.n_states = self.data[0].shape[1]
        self.n_objects = self.data[0].shape[2]
        assert all([sample.shape[2] == self.n_objects for sample in self.data]), \
            "All samples must have the same number of objects."

        self.shuffle = shuffle
        index_sample = np.arange(len(self.data))
        index_per_sample = [np.arange(len(sample)-1) for sample in self.data]
        self.indexes = np.zeros((self.n_samples, 2), dtype=np.int)
        end_prev = 0
        for idx in index_sample:
            start = end_prev
            end = start + len(index_per_sample[idx])
            self.indexes[start:end, 0] = idx
            self.indexes[start:end, 1] = index_per_sample[idx]
            end_prev = end

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
        indexes_y = indexes_X.copy()
        indexes_y[:, 1] = indexes_y[:, 1] + 1

        X = torch.stack([self.data[idx_sample[0]][idx_sample[1]] for idx_sample in indexes_X], dim=0)
        y = torch.stack([self.data[idx_sample[0]][idx_sample[1]] for idx_sample in indexes_y], dim=0)
        y = y[:, VEL]

        return X, y

    def apply_shuffle(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
