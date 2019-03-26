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
                    states.append((i, j, ) + self.room_wh)
        state_to_idx = {s:i for i, s in enumerate(states)}

        self.states = states
        self.states_np = np.array(self.states)

        onehot = np.concatenate([np.eye(len(self.states)), np.repeat(np.array([self.room_wh]), len(self.states), 0)], -1)
        shuffled_states = [states[i] for i in np.random.choice(np.arange(0, len(states)),
                                                               size=len(self.states),
                                                               replace=False)]
        self.states_to_onehot = {s:onehot[i] for i, s in enumerate(shuffled_states)}

        self.states_onehot = np.stack([self.states_to_onehot[x] for x in self.states], 0)

        self.state_to_idx = state_to_idx

        transitions = []
        transitions_onehot = []
        for i, state in enumerate(states):
            next_states = self._gen_transitions(state)
            for ns in next_states:
                transitions.append(list(state) + list(ns) + list(self.room_wh))
                transitions_onehot.append(np.concatenate([self.states_to_onehot[state],
                                                          self.states_to_onehot[tuple(ns.tolist() + list(self.room_wh))],
                                                         ]))
        self.transitions = transitions
        self.transitions_np = np.array(self.transitions)
        self.transitions_onehot = np.array(transitions_onehot)

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
        for state in self.states:
            dist = get_numpy(encoder(from_numpy(self.states_to_onehot[state])).unsqueeze(0))
            Z[state[:2]] = np.argmax(dist) + 1
        return Z

    def sample_states(self, bs):
        return np.stack([self.states_onehot[i, :] for i in np.random.randint(0, len(self.states), bs)], 0)
        #self.states_onehot[np.random.randint(0, len(self.states), bs

    def all_states(self):
        #return np.concatenate([self.states_onehot, np.repeat(np.array([self.room_wh]), len(self.states), 1)])
        return self.states_onehot

class AbstractMDPsContrastive:
    def __init__(self, envs):
        self.envs = [EnvContainer(env) for env in envs]

        self.abstract_dim = 4
        self.state_dim = len(self.envs[0].states) + 2
        self.states = []
        self.state_to_idx = None

        self.encoder = Mlp((128, 128, 128), output_size=self.abstract_dim, input_size=self.state_dim,
                           output_activation=F.softmax, layer_norm=True)
        self.transitions = nn.Parameter(torch.zeros((self.abstract_dim, self.abstract_dim)))

        self.optimizer = optim.Adam(self.encoder.parameters())

    def train(self, max_epochs=100):


        for epoch in range(1, max_epochs + 1):
            stats = self.train_epoch(epoch)

            print(stats)

    def kl(self, dist1, dist2):
        return (dist1 * (torch.log(dist1 + 1e-8) - torch.log(dist2 + 1e-8))).sum(1)

    def entropy(self, dist):
        return -(dist * torch.log(dist + 1e-8)).sum(1)

    def compute_abstract_t(self, env):
        trans = env.transitions_onehot
        #import pdb; pdb.set_trace()
        s1 = trans[:, :self.state_dim]
        s2 = trans[:, self.state_dim:]
        y1 = self.encoder(from_numpy(s1))
        y2 = self.encoder(from_numpy(s2))
        y3 = self.encoder(from_numpy(env.sample_states(s1.shape[0])))

        a_t = from_numpy(np.zeros((self.abstract_dim, self.abstract_dim)))
        for i in range(self.abstract_dim):
            for j in range(self.abstract_dim):
                a_t[i, j] += (y1[:, i] * y2[:, j]).sum(0)

        a_t = a_t / a_t.sum(1)
        return a_t, y1, y2, y3


    def train_epoch(self, epoch):
        stats = OrderedDict([('Loss', 0),
                      ('Converge', 0),
                      ('Diverge', 0),
                      ('Entropy', 0),
                      ('Dev', 0)
                      ])

        data = [self.compute_abstract_t(env) for env in self.envs]
        abstract_t = [x[0] for x in data]
        y1 = torch.cat([x[1] for x in data], 0)
        y2 = torch.cat([x[2] for x in data], 0)
        y3 = torch.cat([x[3] for x in data], 0)

        mean_t = sum(abstract_t) / len(abstract_t)
        dev = [torch.pow(x[0] - mean_t, 2).mean() for x in abstract_t]

        #import pdb; pdb.set_trace()
        #import pdb; pdb.set_trace()
        bs = y1.shape[0]
        l1 = self.kl(y1, y2)
        l2 = self.kl(y2, y1)
        l3 = (-self.kl(y1, y3) - self.kl(y3, y1)) * 20
        #l4 = -self.kl(y3, y1)
        l5 = -self.entropy(y1) * 1
        l6 = sum(dev) / len(dev) * 0
        #import pdb; pdb.set_trace()
        loss = (l1 + l2) + l3  + l5
        #loss = l5
        loss = loss.sum() / bs + l6
        #import pdb; pdb.set_trace()
        loss.backward()
        nn.utils.clip_grad_norm(self.encoder.parameters(), 5.0)
        self.optimizer.step()

        stats['Loss'] += loss.item()
        stats['Converge'] += ((l1 + l2).sum() / bs ).item()
        stats['Diverge'] += (l3.sum() / bs).item()
        stats['Entropy'] += (l5.sum() / bs).item()
        stats['Dev'] += l6.item()

        self.y1 = y1

        return stats

    def gen_plot(self):
        plots = [env.gen_plot(self.encoder) for env in self.envs]

        plots = np.concatenate(plots, 1)

        plt.imshow(plots)
        plt.savefig('/home/jcoreyes/abstract/rlkit/examples/abstractmdp/fig1.png')
        plt.show()


if __name__ == '__main__':
    # laplacian = Laplacian(FourRoomsModEnv())
    # vals, vectors= laplacian.generate_laplacian()
    # laplacian.gen_plot(vectors[:, 1])
    envs = [FourRoomsModEnv(gridsize=15, room_wh=(6, 6)),
            FourRoomsModEnv(gridsize=15, room_wh=(7, 7)),
            #FourRoomsModEnv(gridsize=15, room_wh=(6, 7)),
            ]
    a = AbstractMDPsContrastive(envs)
    a.train(max_epochs=200)
    print(a.y1)
    a.gen_plot()


