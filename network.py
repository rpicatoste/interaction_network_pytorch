import torch
import torch.nn as nn

from pytorch_commons import device

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
        """
        Args:
            x: [batch_size, input_size, n_relations]
        Returns:
            [batch_size, effect_dim, n_relations]
        """
        batch_size, _, n_relations = B.shape

        # f_R is applied to each relation independently, and therefore to apply f_R it doesn't
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
        """
        Args:
            C: [batch_size,  input_size = D_s + D_x + D_e, n_objects]
        Returns:
            [batch_size * n_objects, 2] speedX and speedY
        """
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
