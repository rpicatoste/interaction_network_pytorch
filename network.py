import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import random

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

from physics_engine import POS_X, POS_Y, STATE, VEL, POS

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

        forces_first_cuda_action_done = True


def get_batch(data, batch_size, n_objects, relation_dim, shuffle=True):
    n_relations = n_objects * (n_objects - 1)  # number of edges in fully connected graph

    if shuffle:
        rand_idx = [random.randint(0, len(data) - 2) for _ in range(batch_size)]
    else:
        rand_idx = list(range(0, len(data) - 2))
        batch_size = len(rand_idx)

    label_idx = [idx + 1 for idx in rand_idx]

    batch_data = data[rand_idx]
    label_data = data[label_idx]

    objects = batch_data[:, :, STATE]

    # receiver_relations, sender_relations - onehot encoding matrices
    # each column indicates the receiver and sender object’s index

    receiver_relations = np.zeros((batch_size, n_objects, n_relations), dtype=float)
    sender_relations = np.zeros((batch_size, n_objects, n_relations), dtype=float)

    cnt = 0
    for i in range(n_objects):
        for j in range(n_objects):
            if (i != j):
                receiver_relations[:, i, cnt] = 1.0
                sender_relations[:, j, cnt] = 1.0
                cnt += 1

    # There is no relation info in solar system task, just fill with zeros
    relation_info = np.zeros((batch_size, n_relations, relation_dim))
    target = label_data[:, :, VEL]

    objects = Variable(torch.FloatTensor(objects))
    sender_relations = Variable(torch.FloatTensor(sender_relations))
    receiver_relations = Variable(torch.FloatTensor(receiver_relations))
    relation_info = Variable(torch.FloatTensor(relation_info))
    target = Variable(torch.FloatTensor(target)).view(-1, 2)

    objects = objects.to(device)
    sender_relations = sender_relations.to(device)
    receiver_relations = receiver_relations.to(device)
    relation_info = relation_info.to(device)
    target = target.to(device)

    return objects, sender_relations, receiver_relations, relation_info, target


# Relation-centric Nerural Network
# This NN takes all information about relations in the graph and outputs effects of all interactions
# between objects.
class RelationalModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(RelationalModel, self).__init__()

        self.output_size = output_size

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )

    def forward(self, x):
        '''
        Args:
            x: [batch_size, n_relations, input_size]
        Returns:
            [batch_size, n_relations, output_size]
        '''
        batch_size, n_relations, input_size = x.size()
        x = x.view(-1, input_size)
        x = self.layers(x)
        x = x.view(batch_size, n_relations, self.output_size)
        return x

# Object-centric Neural Network
# This NN takes information about all objects and effects on them, then outputs prediction of the
# next state of the graph.
class ObjectModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ObjectModel, self).__init__()

        self.input_size = input_size

        self.layers = nn.Sequential(
            nn.Linear(self.input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),  # speedX and speedY
        )

    def forward(self, x):
        '''
        Args:
            x: [batch_size, n_objects, input_size]
        Returns:
            [batch_size * n_objects, 2] speedX and speedY
        '''
        # input_size = x.size(2)
        self.n_objects = x.size(1)
        x = x.view(-1, self.input_size)

        return self.layers(x)


# Interaction Network
# IN involves only matrix operations that do not contain learnable parameters.
class InteractionNetwork(nn.Module):
    def __init__(self, n_objects, n_features, relation_dim, effect_dim):
        super(InteractionNetwork, self).__init__()

        self.relational_model = RelationalModel(input_size=2 * n_features + relation_dim,
                                                output_size=effect_dim,
                                                hidden_size=150)
        self.object_model = ObjectModel(input_size=n_features + effect_dim,
                                        hidden_size=100)

        self.to(device)

    def forward(self, objects, sender_relations, receiver_relations, relation_info, X=None):
        B = self.marshalling(objects, sender_relations, receiver_relations, relation_info)
        E = self.relational_model(B)

        C = self.aggregator(objects, receiver_relations, E, X=X)
        predicted = self.object_model(C)

        return predicted

    @staticmethod
    def marshalling(objects, sender_relations, receiver_relations, relation_info):
        senders = sender_relations.permute(0, 2, 1).bmm(objects)
        receivers = receiver_relations.permute(0, 2, 1).bmm(objects)

        return torch.cat([senders, receivers, relation_info], 2)

    @staticmethod
    def aggregator(objects, receiver_relations, E, X=None):
        effect_receivers = receiver_relations.bmm(E)
        if X is None:
            C = torch.cat([objects, effect_receivers], dim=2)
        else:
            C = torch.cat([objects, X, effect_receivers], dim=2)

        return C


class Model:
    def __init__(self, n_objects, n_features, relation_dim, effect_dim):
        self.network = InteractionNetwork(n_objects, n_features, relation_dim, effect_dim)
        self.optimizer = optim.Adam(self.network.parameters())
        self.criterion = nn.MSELoss()
        self.losses = []
        self.n_objects = n_objects
        self.relation_dim = relation_dim
        self.n_relations = self.n_objects * (self.n_objects - 1)  # number of e

    def train(self, n_epoch, batches_per_epoch, data):
        print('Training ...')
        for epoch in range(n_epoch):
            for _ in range(batches_per_epoch):
                objects, sender_relations, receiver_relations, relation_info, target = \
                    get_batch(data=data,
                              batch_size=30,
                              n_objects=self.n_objects,
                              relation_dim=self.relation_dim)
                predicted = self.network(objects,
                                         sender_relations,
                                         receiver_relations,
                                         relation_info)
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


    def test(self, data, dt, rel_dim):

        objects, sender_relations, receiver_relations, relation_info, _ = \
            get_batch(data=data,
                      batch_size=len(data),
                      n_objects=self.n_objects,
                      relation_dim=rel_dim,
                      shuffle=False)

        print('self.n_objects', self.n_objects)
        print('sender_relations.shape', sender_relations.shape)
        sender_relations_1 = sender_relations#[0:1, :, :]
        receiver_relations_1 = receiver_relations#[0:1, :, :]
        relation_info_1 = relation_info#[0:1, :, :]
        print('sender_relations_1.shape', sender_relations_1.shape)

        n_steps = len(objects)

        # Preallocate space for pos and velocity and get the first value.
        pos_predictions = torch.zeros_like(objects[:, :, POS])
        speed_predictions = torch.zeros_like(objects[:, :, VEL])
        print(data.shape)
        print(type(data))

        prev_state = objects[0:1]

        with torch.no_grad():
            for step_i in range(1):#n_steps):
                print('prev_state.shape', prev_state.shape)
                print('objects.shape', objects.shape)
                print('sender_relations_1.shape', sender_relations_1.shape)
                speed_prediction = self.network(objects,
                                                sender_relations_1,
                                                receiver_relations_1,
                                                relation_info_1)
                print('speed_prediction.shape', speed_prediction.shape)
                pos_prediction = prev_state[0, :, POS] + speed_prediction * dt

                speed_predictions[step_i] = speed_prediction
                pos_predictions[step_i] = pos_prediction

                prev_state[0, :, POS] = pos_prediction
                prev_state[0, :, VEL] = speed_prediction

        object_i = 1
        plt.plot(data[:, object_i, POS_X],
                 data[:, object_i, POS_Y],
                 label='real')
        plt.plot(pos_predictions[:, object_i, POS_X],
                 pos_predictions[:, object_i, POS_Y],
                 label='pred')
