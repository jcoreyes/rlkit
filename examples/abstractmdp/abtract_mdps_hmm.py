import numpy as np


import gym
import numpy as np
import gym_minigrid
from gym_minigrid.envs.fourrooms import FourRoomsModEnv, BridgeEnv, WallEnv
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from rlkit.torch.networks import Mlp
from rlkit.torch.pytorch_util import from_numpy, get_numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from collections import OrderedDict


class EnvContainer:
    def __init__(self, env):
        self.env = env
        self.width = self.env.grid.height
        self.height = self.env.grid.height
        self.room_wh = self.env.room_wh

        states = []
        for j in range(self.env.grid.height):
            for i in range(self.env.grid.width):
                if self.env.grid.get(i, j) == None:
                    states.append((i, j))
        state_to_idx = {s:i for i, s in enumerate(states)}

        self.states = states
        self.states_np = np.array(self.states)
        self.state_to_idx = state_to_idx

        transitions = []
        for i, state in enumerate(states):
            next_states = self._gen_transitions(state)
            for ns in next_states:
                transitions.append([state_to_idx[state],state_to_idx[tuple(ns.tolist())]])
        self.transitions = transitions
        self.transitions_np = np.array(self.transitions)

    def _gen_transitions(self, state):

        actions = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        next_states = []
        for action in actions:
            ns = np.array(state)[:2] + action
            if ns[0] >= 0 and ns[1] >= 0 and ns[0] < self.width and ns[1] < self.height and \
                self.env.grid.get(*ns) == None:
                next_states.append(ns)
        return next_states


    def gen_plot(self, encoder):

        #X = np.arange(0, self.width)
        #Y = np.arange(0, self.height)
        #X, Y = np.meshgrid(X, Y)
        Z = np.zeros((self.width, self.height))
        for i, state in enumerate(self.states):
            dist = encoder[i, :]
            Z[state[:2]] = np.argmax(dist) + 1
        return Z

    def sample_states(self, bs):
        return self.states_np[np.random.randint(0, len(self.states), bs), :]

    def all_states(self):
        return self.states_np

class AbstractMDPsContrastive:
    def __init__(self, envs):
        self.envs = [EnvContainer(env) for env in envs]

        self.n_abstract_mdps = 2
        self.abstract_dim = 4
        self.state_dim = 4
        self.states = []
        self.state_to_idx = None

        self.encoder = Mlp((64, 64, 64), output_size=self.abstract_dim, input_size=self.state_dim,
                           output_activation=F.softmax, layer_norm=True)
        self.transitions = nn.Parameter(torch.zeros((self.abstract_dim, self.abstract_dim)))

        self.optimizer = optim.Adam(self.encoder.parameters())

    def train(self, max_epochs=100):

        # Initialize A and B
        n_states = len(self.envs[0].states)
        A = np.ones((self.abstract_dim, self.abstract_dim))
        for i in range(self.abstract_dim):
            A[i, i] += 10
        A /= A.sum(1, keepdims=True)

        B = np.ones((n_states, self.abstract_dim)) / n_states
        #B = np.random.uniform(size=(n_states, self.abstract_dim))
        B /= B.sum(0, keepdims=True)
        O = np.array(self.envs[0].transitions)

        num_seq = O.shape[0]

        alpha = np.zeros((num_seq, 2, self.abstract_dim))
        beta = np.zeros((num_seq, 2, self.abstract_dim))
        pi = np.ones(self.abstract_dim) / 4

        for epoch in range(1, max_epochs + 1):
            # E step
            alpha[:, 0, :] = pi * B[O[:, 0], :]
            alpha[:, 1, :] = 0
            for j in range(self.abstract_dim):
                for i in range(self.abstract_dim):
                    alpha[:, 1, j] += alpha[:, 0, i] * A[i, j] * B[O[:, 1], j]

            beta[:, 1, :] = 1
            beta[:, 0, :] = 0
            for i in range(self.abstract_dim):
                for j in range(self.abstract_dim):
                    beta[:, 0, i] += A[i, j] * B[O[:, 1], j] * beta[:, 1, j]
            #import pdb; pdb.set_trace()
            # Normalize to deal with underflow
            alpha /= alpha.sum(-1, keepdims=True)
            beta /= beta.sum(-1, keepdims=True)

            gamma = alpha * beta
            # normalize gamma
            gamma = gamma / gamma.sum(-1, keepdims=True)

            zeta = np.zeros((num_seq, self.abstract_dim, self.abstract_dim))
            #for i in range(self.abstract_dim):
            #    zeta[:, 0, i, :] = alpha[:, 0, i].reshape((num_seq, 1)) * A[i, :] * B[O[:, 1], :] * beta[:, 1, :] / (alpha[:, 0, :] * beta[:, 0, :]).sum(keepdims=True)
            norm = (alpha[:, 0, :] * beta[:, 0, :]).sum(-1)
            for i in range(self.abstract_dim):
                for j in range(self.abstract_dim):
                    zeta[:, i, j] = alpha[:, 0, i] * A[i, j] * B[O[:, 1], j] * beta[:, 1, j] / norm
            #import pdb; pdb.set_trace()

            #zeta /= zeta.sum(-1, keepdims=True)

            # M Step
            for i in range(self.abstract_dim):
                for j in range(self.abstract_dim):
                    A[i, j] = zeta[:, i, j].sum() / zeta[:, i, :].sum()
                    #A[i, j] = (zeta[:, 0, i, j] / zeta[:, 0, i, :].sum(-1)).mean()

            #import pdb; pdb.set_trace()
            norm = gamma.sum(1)
            for k in range(n_states):
                mask = O == k
                mask = np.expand_dims(mask, -1)
                B[k, :] = (gamma * mask).sum(1).sum(0) / gamma.sum(1).sum(0)
                #import pdb; pdb.set_trace()
                #B[k, :] = ((gamma * mask).sum(1) / norm).mean(0)
                #tmp = ((gamma * mask).sum(1) / norm)
                #tmp /= tmp.sum(0) + 1e-12
                #B[k, :] = tmp.sum(0)
            #import pdb; pdb.set_trace()
            #B /= B.sum(1, keepdims=True)

            #print(B[:5])
        #import pdb; pdb.set_trace()

        B /= B.sum(1, keepdims=True)
        print(A)
        print(B[:5])
        self.encoder = B


    def kl(self, dist1, dist2):
        return (dist1 * (torch.log(dist1 + 1e-8) - torch.log(dist2 + 1e-8))).sum(1)

    def entropy(self, dist):
        return -(dist * torch.log(dist + 1e-8)).sum(-1)


    def gen_plot(self):
        plots = [env.gen_plot(self.encoder) for env in self.envs]

        plots = np.concatenate(plots, 1)

        plt.imshow(plots)
        #plt.savefig('/home/jcoreyes/abstract/rlkit/examples/abstractmdp/fig1.png')
        plt.show()


if __name__ == '__main__':
    # laplacian = Laplacian(FourRoomsModEnv())
    # vals, vectors= laplacian.generate_laplacian()
    # laplacian.gen_plot(vectors[:, 1])
    envs = [#FourRoomsModEnv(gridsize=15, room_wh=(6, 6)),
            FourRoomsModEnv(gridsize=15, room_wh=(7, 7)),
            #FourRoomsModEnv(gridsize=15, room_wh=(7, 7), close_doors=["west"])
            #FourRoomsModEnv(gridsize=15, room_wh=(6, 7)),
            ]
    a = AbstractMDPsContrastive(envs)
    a.train(max_epochs=200)
    #print(a.mean_t)
    #print(a.y1)
    a.gen_plot()

