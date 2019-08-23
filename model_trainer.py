import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

from network import InteractionNetwork
from physics_engine import POS_X, POS_Y, VEL, POS, COLOR_LIST, MASS
from pytorch_commons import device


class ModelTrainer:

    def __init__(self, n_objects, state_dim, relation_dim, effect_dim, output_dim):
        self.network = InteractionNetwork(n_objects, state_dim, relation_dim, effect_dim, output_dim)

        self.losses = []
        self.n_objects = n_objects
        self.relation_dim = relation_dim

        self.optimizer = optim.Adam(self.network.parameters())
        self.criterion = nn.MSELoss()

    def train(self, n_epoch, data_loader):
        new_n_objects = data_loader.n_objects
        self.network.update_matrices(new_n_objects)

        print('Training ...')
        for epoch in range(n_epoch):
            for objects, targets in data_loader:
                # objects, targets = get_batch(data_loader._data, data_loader.batch_size)
                predicted = self.network(objects)
                loss = self.criterion(predicted, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.losses.append(np.sqrt(loss.item()))

                # print(loss)
                # print('predicted', predicted.shape)
                # print('objects', objects.shape)
                # print('target', targets.shape)
                # print('objects', objects[:3, VEL, 0])
                # # print('predicted', predicted[:3])
                # print('target', targets[:3, :, 0])


            if epoch % (n_epoch/20) == 0:
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

        states, _ = data_loader[0]

        # Preallocate space for pos and velocity and get the first value.
        next_states_hat = torch.zeros_like(states)

        prev_state = states[0:1]

        with torch.no_grad():
            for step_i in range(n_steps-1):
                # print('prev_state.shape', prev_state.shape)
                # print('objects.shape', objects.shape)
                speed_hat = self.network(prev_state)
                # print('speed_prediction.shape', speed_prediction.shape)
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
