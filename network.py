import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import random

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

from physics_engine import POS_X, POS_Y, STATE, VEL, POS, COLOR_LIST

device = 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cuda':
    if 'forces_first_cuda_action_done' not in locals():

        try:
            a = torch.rand((2, 3, 2)).to('cuda')
            b = torch.rand((2, 3, 4)).to('cuda')
            a = a.permute(0, 2, 1).bmm(b)
        except:
            print('Forced first action in cuda to avoid later error.')

        torch.Tensor.ndim = property(lambda x: len(x.shape))  # so tensors can be plot
        forces_first_cuda_action_done = True


def get_batch(data, batch_size, shuffle=True):
    if shuffle:
        rand_idx = [random.randint(0, len(data) - 2) for _ in range(batch_size)]
    else:
        rand_idx = list(range(0, len(data) - 2))

    label_idx = [idx + 1 for idx in rand_idx]

    batch_data = data[rand_idx]
    label_data = data[label_idx]

    objects = batch_data[:, STATE, :]
    target = label_data[:, VEL, :]

    objects = Variable(torch.FloatTensor(objects))
    target = Variable(torch.FloatTensor(target))

    objects = objects.to(device)
    target = target.to(device)

    return objects, target


# Relation-centric Nerural Network
# This NN takes all information about relations in the graph and outputs effects of all interactions
# between objects.
class RelationalModel(nn.Module):
    def __init__(self, input_dim, effect_dim, hidden_size):
        super().__init__()

        self.input_dim = input_dim
        self.D_e = effect_dim

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, effect_dim),
            nn.ReLU()
        )

    def forward(self, B):
        '''
        Args:
            x: [batch_size, input_size, n_relations]
        Returns:
            [batch_size, effect_dim, n_relations]
        '''
        batch_size, _, n_relations = B.shape

        # f_R es applied to each relation independently, and therefore to apply f_R it doesn't
        # matter if it comes from different relations in a sample o from different samples in the
        # batch.
        x = B.permute(0, 2, 1).contiguous().view(-1, self.input_dim)
        x = self.layers(x)

        # Reshape the output to have the outputs of the relations for each sample.
        x = x.view(batch_size, n_relations, self.D_e).permute(0, 2, 1)

        return x

# Object-centric Neural Network
# This NN takes information about all objects and effects on them, then outputs prediction of the
# next state of the graph.
class ObjectModel(nn.Module):

    def __init__(self, input_dim, hidden_size, output_dim):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),  # speedX and speedY
        )

    def forward(self, C):
        '''
        Args:
            C: [batch_size,  input_size = D_s + D_x + D_e, n_objects]
        Returns:
            [batch_size * n_objects, 2] speedX and speedY
        '''
        # input_size = x.size(2)
        batch_size, _, n_objects = C.shape

        x = C.permute(0, 2, 1).contiguous().view(-1, self.input_dim)
        x = x.view(-1, self.input_dim)
        x = self.layers(x)
        x = x.view(batch_size, n_objects, self.output_dim).permute(0, 2, 1)

        return x


# Interaction Network
# IN involves only matrix operations that do not contain learnable parameters.
class InteractionNetwork(nn.Module):

    def __init__(self, n_objects, state_dim, relation_dim, effect_dim, output_dim):
        super().__init__()

        self.N_O = n_objects
        self.D_s = state_dim
        self.D_r = relation_dim
        self.D_e = effect_dim
        self.D_p = output_dim

        self.N_R, self.R_r, self.R_s, self.R_a = self.generate_matrices(self.N_O)


        self.f_R = RelationalModel(input_dim=2 * self.D_s + self.D_r,
                                   effect_dim=self.D_e,
                                   hidden_size=150)

        self.f_O = ObjectModel(input_dim=self.D_s + self.D_e,
                               hidden_size=100,
                               output_dim=self.D_p)

        self.to(device)

    def update_matrices(self, N_O):
        self.N_O = N_O
        self.N_R, self.R_r, self.R_s, self.R_a = self.generate_matrices(self.N_O)

    def generate_matrices(self, N_O):
        N_R = N_O * (N_O - 1)
        R_r = torch.zeros((N_O, N_R), dtype=torch.float).to(device)
        R_s = torch.zeros((N_O, N_R), dtype=torch.float).to(device)
        cnt = 0
        for ii in range(N_O):
            for jj in range(N_O):
                if (ii != jj):
                    R_r[ii, cnt] = 1.0
                    R_s[jj, cnt] = 1.0
                    cnt += 1

        R_a = torch.zeros((self.D_r, N_R), dtype=torch.float).to(device)

        return N_R, R_r, R_s, R_a

    def forward(self, O, X=None):
        if X is None:
            X = torch.zeros((0, self.N_O)).to(device)
        self.D_x = 0

        B = self.marshalling(O)
        E = self.f_R(B)
        C = self.aggregator(objects=O, E=E, X=X)
        predicted = self.f_O(C)

        assert self.D_s + self.D_x + self.D_e == C.shape[1]
        assert self.N_O == C.shape[2]

        # print('O.shape', O.shape)
        # print('x', X.shape)
        # print('B.shape', B.shape)
        # print('E.shape', E.shape)
        # print('C.shape', C.shape)
        # print('predicted.shape', predicted.shape)

        return predicted

    def marshalling(self, O):
        batch_size = O.shape[0]
        assert self.R_s.shape[0] == self.N_O
        assert self.R_r.shape[0] == self.N_O
        assert O.shape[-1] == self.N_O

        O_R_r = O @ self.R_r
        O_R_s = O @ self.R_s
        B = torch.cat([O_R_r,
                       O_R_s,
                       self.R_a.expand(batch_size, -1, -1)],
                      dim=len(O_R_r.shape)-2)

        return B

    def aggregator(self, objects, E, X):
        effect_receivers = E @ self.R_r.t()
        C = torch.cat([objects, effect_receivers], dim=len(objects.shape)-2)

        return C


class Model:
    def __init__(self, n_objects, state_dim, relation_dim, effect_dim, output_dim):
        self.network = InteractionNetwork(n_objects, state_dim, relation_dim, effect_dim, output_dim)

        self.losses = []
        self.n_objects = n_objects
        self.relation_dim = relation_dim

        self.optimizer = optim.Adam(self.network.parameters())
        self.criterion = nn.MSELoss()

    def train(self, n_epoch, batches_per_epoch, data):
        print('Training ...')
        for epoch in range(n_epoch):
            for _ in range(batches_per_epoch):
                objects, target = get_batch(data=data, batch_size=1000)

                predicted = self.network(objects)
                # print('predicted', predicted.shape)
                # print('target', target.shape)
                loss = self.criterion(predicted, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.losses.append(np.sqrt(loss.item()))

            if epoch % (n_epoch/20) == 0:
                plt.figure(figsize=(10, 5))
                plt.title('Epoch %s RMS Error %s' % (epoch, np.sqrt(np.mean(self.losses[-100:]))))
                plt.plot(self.losses)
                clear_output(True)
                print(f'Done epoch {epoch}... ')
                plt.show()


    def test(self, test_data, dt):

        objects, _ = get_batch(data=test_data, batch_size=len(test_data), shuffle=False)

        n_steps = len(objects)
        n_objects = len(objects[0, 0])

        # Preallocate space for pos and velocity and get the first value.
        pos_predictions = torch.zeros_like(objects[:, POS, :])
        speed_predictions = torch.zeros_like(objects[:, VEL, :])
        print(test_data.shape)
        print(type(test_data))

        prev_state = objects[0:1]

        with torch.no_grad():
            for step_i in range(n_steps):
                # print('prev_state.shape', prev_state.shape)
                # print('objects.shape', objects.shape)
                speed_prediction = self.network(prev_state)
                # print('speed_prediction.shape', speed_prediction.shape)
                pos_prediction = prev_state[0, POS, :] + speed_prediction * dt

                speed_predictions[step_i] = speed_prediction
                pos_predictions[step_i] = pos_prediction

                prev_state[0, POS, :] = pos_prediction
                prev_state[0, VEL, :] = speed_prediction

        plt.figure()
        for object_i in range(n_objects):
            modifier_idx = object_i % len(COLOR_LIST)
            plt.plot(test_data[:, POS_X, object_i],
                     test_data[:, POS_Y, object_i],
                     COLOR_LIST[modifier_idx],
                     label=f'real {object_i}')
            plt.plot(pos_predictions[:, 0, object_i].cpu(),
                     pos_predictions[:, 1, object_i].cpu(),
                     '--' + COLOR_LIST[modifier_idx],
                     label=f'pred {object_i}',
                     )
        plt.legend()
