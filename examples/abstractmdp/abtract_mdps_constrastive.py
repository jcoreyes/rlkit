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

class AbstractMDPContrastive:
    def __init__(self, env):
        self.env = env
        self.width = self.env.grid.height
        self.height = self.env.grid.height
        self.abstract_dim = 4
        self.state_dim = 2
        self.states = []
        self.state_to_idx = None

        self.encoder = Mlp((64, 64, 64), output_size=self.abstract_dim, input_size=self.state_dim,
                           output_activation=F.softmax, layer_norm=True)

        states = []
        for j in range(self.env.grid.height):
            for i in range(self.env.grid.width):
                if self.env.grid.get(i, j) == None:
                    states.append((i, j))
        state_to_idx = {s:i for i, s in enumerate(states)}

        self.states = states
        self.state_to_idx = state_to_idx

        transitions = []
        for i, state in enumerate(states):
            next_states = self._gen_transitions(state)
            for ns in next_states:
                transitions.append(list(state) + list(ns))
        self.transitions = transitions

        self.optimizer = optim.Adam(self.encoder.parameters())

    def _gen_transitions(self, state):

        actions = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        next_states = []
        for action in actions:
            ns = np.array(state) + action
            if ns[0] >= 0 and ns[1] >= 0 and ns[0] < self.width and ns[1] < self.height and \
                self.env.grid.get(*ns) == None:
                next_states.append(ns)
        return next_states


    def gen_plot(self):

        #X = np.arange(0, self.width)
        #Y = np.arange(0, self.height)
        #X, Y = np.meshgrid(X, Y)
        Z = np.zeros((self.width, self.height))
        for state in self.states:
            dist = get_numpy(self.encoder(from_numpy(np.array(state)).unsqueeze(0)))
            Z[state] = np.argmax(dist) + 1

        #fig = plt.figure()
        #ax = Axes3D(fig)
        #surf = ax.plot_surface(X, Y, Z)
        #cset = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
        #ax.clabel(cset)
        plt.imshow(Z)
        plt.show()

    def train(self, max_epochs=100):
        transitions = from_numpy(np.array(self.transitions))
        dataset = data.TensorDataset(transitions[:, :2], transitions[:, 2:])
        dataloader = data.DataLoader(dataset, batch_size=32, shuffle=True)

        for epoch in range(1, max_epochs + 1):
            stats = self.train_epoch(dataloader, epoch)

            print(stats)

    def kl(self, dist1, dist2):
        return (dist1 * (torch.log(dist1 + 1e-8) - torch.log(dist2 + 1e-8))).sum(1)

    def entropy(self, dist):
        return -(dist * torch.log(dist + 1e-8)).sum(1)

    def train_epoch(self, dataloader, epoch):
        stats = dict([('Loss', 0)])
        for batch_idx, (s1, s2) in enumerate(dataloader):
            bs = s1.shape[0]
            self.optimizer.zero_grad()

            s3 = np.array([self.states[x] for x in np.random.randint(0, len(self.states), bs)])

            y1 = self.encoder(s1)
            y2 = self.encoder(s2)
            y3 = self.encoder(from_numpy(s3))

            l1 = self.kl(y1, y2)
            l2 = self.kl(y2, y1)
            l3 = -self.kl(y1, y3)
            #l4 = -self.kl(y3, y1)
            l5 = -self.entropy(y1)

            loss = (l1 + l2) + 0.8*l3 +  0.3*l5
            loss = loss.sum() / bs
            loss.backward()
            nn.utils.clip_grad_norm(self.encoder.parameters(), 5.0)
            self.optimizer.step()

            stats['Loss'] += loss.item()
        stats['Loss'] /= (batch_idx + 1)
        return stats

if __name__ == '__main__':
    # laplacian = Laplacian(FourRoomsModEnv())
    # vals, vectors= laplacian.generate_laplacian()
    # laplacian.gen_plot(vectors[:, 1])
    env = FourRoomsModEnv(gridsize=15, room_wh=(6, 6))
    a = AbstractMDPContrastive(env)
    a.train(max_epochs=200)
    a.gen_plot()


