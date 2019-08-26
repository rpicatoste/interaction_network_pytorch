import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from pathlib import Path

from network import InteractionNetwork
from physics_engine import POS_X, POS_Y, VEL, POS, COLOR_LIST, MASS
from pytorch_commons import device


DEFAULT_FILE_NAME = 'interaction_network'


class ModelTrainer:

    def __init__(self, *args, **kwargs):

        self.args, self.kwargs = args, kwargs
        self.network = InteractionNetwork(*args, **kwargs)

        self.losses = []

        self.optimizer = optim.Adam(self.network.parameters())
        self.criterion = nn.MSELoss()

        self.scaler = None

    def save(self, file_path=None, time_stamped=True):
        if file_path is None:
            file_path = DEFAULT_FILE_NAME

        file_path = Path(file_path).with_suffix('')

        if time_stamped:
            timestr = time.strftime("%Y_%m_%d-%H%M%S")
            file_path = Path(str(file_path) + '_' + timestr)
        print(f'Saving {file_path}')

        torch.save({
            'model_args': self.args,
            'model_kwargs': self.kwargs,
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': np.array(self.losses, dtype=np.float32),
            'scaler': self.scaler,  # Needed for inference.
        }, file_path.with_suffix('.tar'))

    @staticmethod
    def load(file_path=None):
        if file_path is None:
            file_path = find_biggest_version(DEFAULT_FILE_NAME)

        print(f'Loading {file_path}')

        checkpoint = torch.load(file_path)
        model = ModelTrainer(*checkpoint['model_args'], **checkpoint['model_kwargs'])
        model.network.load_state_dict(checkpoint['network_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.losses = checkpoint['losses'].tolist()
        model.scaler = checkpoint['scaler']

        return model

    def train(self, n_epoch, data_loader):
        self.scaler = data_loader.scaler

        new_n_objects = data_loader.n_objects
        self.network.update_matrices(new_n_objects)

        self.network.train()

        print('Training ...')
        for epoch in range(n_epoch):
            for objects, targets in data_loader:

                predicted = self.network(objects)
                loss = self.criterion(predicted, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.losses.append(np.float32(loss.item()))

            if epoch % (n_epoch/5) == 0:
                plt.figure(figsize=(10, 5))
                plt.title('Epoch %s RMS Error %s' % (epoch, np.sqrt(np.mean(self.losses[-100:]))))
                plt.plot(self.losses)
                clear_output(True)
                print(f'Done epoch {epoch}... ')
                plt.show()

    def test(self, data_loader, dt):

        n_steps = data_loader.batch_size
        n_objects = data_loader.n_objects
        self.network.update_matrices(n_objects)

        self.network.eval()

        states, _ = data_loader[0]

        # Preallocate space for pos and velocity and get the first value.
        next_states_hat = torch.zeros_like(states)

        prev_state = states[0:1]

        with torch.no_grad():
            for step_i in range(n_steps-1):
                speed_hat = self.network(prev_state)

                next_state = torch.zeros_like(prev_state).to(device)
                next_state[0, MASS, :] = prev_state[0, MASS, :]
                next_state[0, POS, :] = prev_state[0, POS, :] + speed_hat * dt
                next_state[0, VEL, :] = speed_hat

                next_states_hat[step_i] = next_state
                prev_state = next_state

        plt.figure(figsize=(10, 10))
        for object_i in range(n_objects):
            modifier_idx = object_i % len(COLOR_LIST)
            plt.plot(states[:, POS_X, object_i].cpu(),
                     states[:, POS_Y, object_i].cpu(),
                     COLOR_LIST[modifier_idx],
                     label=f'real {object_i}')
            plt.plot(next_states_hat[:, POS_X, object_i].cpu(),
                     next_states_hat[:, POS_Y, object_i].cpu(),
                     '--o' + COLOR_LIST[modifier_idx],
                     markersize=1.5,
                     label=f'pred {object_i}',
                     )
        plt.legend()

        return next_states_hat


def find_biggest_version(file_path, extension='.tar', pattern=''):
    file_path = str(file_path)

    file_paths = [f for f in os.listdir() if all([file_path in f,
                                                  pattern in f,
                                                  extension in f])]
    return Path(max(file_paths))
