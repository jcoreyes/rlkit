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
        self.state_to_idx = state_to_idx

        transitions = []
        for i, state in enumerate(states):
            next_states = self._gen_transitions(state)
            for ns in next_states:
                transitions.append(list(state) + list(ns) + list(self.room_wh))
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
        for state in self.states:
            dist = get_numpy(encoder(from_numpy(np.array(state)).unsqueeze(0)))
            Z[state[:2]] = np.argmax(dist) + 1
        return Z

    def sample_states(self, bs):
        return self.states_np[np.random.randint(0, len(self.states), bs), :]

    def all_states(self):
        return np.concatenate([self.states_np, np.repeat(np.array([self.room_wh]), len(self.states), 1)])

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

        data_lst = []
        for i, env in enumerate(self.envs):
            d = np.array(env.transitions)
            d = np.concatenate([d, np.zeros((d.shape[0], 1)) + i], 1)
            data_lst.append(d)

        all_data = from_numpy(np.concatenate(data_lst, 0))

        dataset = data.TensorDataset(all_data)
        dataloader = data.DataLoader(dataset, batch_size=32, shuffle=True)

        mixture = from_numpy(np.ones((len(self.envs), self.n_abstract_mdps)) / self.n_abstract_mdps)
        all_abstract_t = from_numpy(np.ones((self.n_abstract_mdps, self.abstract_dim, self.abstract_dim)) / self.abstract_dim)
        for epoch in range(1, max_epochs + 1):
            stats = self.train_epoch(dataloader, epoch, mixture, all_abstract_t)

            print(stats)

    def kl(self, dist1, dist2):
        return (dist1 * (torch.log(dist1 + 1e-8) - torch.log(dist2 + 1e-8))).sum(1)

    def entropy(self, dist):
        return -(dist * torch.log(dist + 1e-8)).sum(-1)

    def compute_abstract_t(self, env):
        trans = env.transitions_np
        s1 = trans[:, :self.state_dim]
        s2 = trans[:, self.state_dim:]
        y1 = self.encoder(from_numpy(s1))
        y2 = self.encoder(from_numpy(s2))
        y3 = self.encoder(from_numpy(env.sample_states(s1.shape[0])))

        a_t = from_numpy(np.zeros((self.abstract_dim, self.abstract_dim)))
        for i in range(self.abstract_dim):
            for j in range(self.abstract_dim):
                a_t[i, j] += (y1[:, i] * y2[:, j]).sum(0)

        n_a_t = from_numpy(np.zeros((self.abstract_dim, self.abstract_dim)))
        for i in range(self.abstract_dim):
            n_a_t[i, :] = a_t[i, :] / a_t[i, :].sum()


        return n_a_t, y1, y2, y3


    def train_epoch(self, dataloader, epoch, mixture, all_abstract_t):
        stats = OrderedDict([('Loss', 0),
                      ('Converge', 0),
                      ('Diverge', 0),
                      ('Entropy1', 0),
                             ('Entropy2', 0),
                      ('Dev', 0)
                      ])

        data = [self.compute_abstract_t(env) for env in self.envs]
        abstract_t = [x[0] for x in data]

        for i in range(self.n_abstract_mdps):
            all_abstract_t[i, :, :] = sum([mixture[j, i] * abstract_t[j] for j in range(len(self.envs))])

        # compute likelihood of mdp's abstract transition under all current abstract transitions
        new_mixture = from_numpy(np.zeros(mixture.shape))
        for j in range(len(abstract_t)):
            for i in range(self.n_abstract_mdps):
                a = abstract_t[j]
                dev = torch.pow(all_abstract_t[i, :, :] - a, 2).mean()
                new_mixture[j, i] = dev
        new_mixture = new_mixture / new_mixture.sum(1, keepdim=True)

        y1 = torch.cat([x[1] for x in data], 0)
        y2 = torch.cat([x[2] for x in data], 0)
        y3 = torch.cat([x[3] for x in data], 0)

        sample = from_numpy(np.random.randint(0, y1.shape[0], 128)).long()
        y1 = y1[sample]
        y2 = y2[sample]
        y3 = y3[sample]

        #mean_t = sum(abstract_t) / len(abstract_t)
        #self.mean_t = mean_t
        #dev = [torch.pow(x[0] - mean_t, 2).mean() for x in abstract_t]


        #sample = from_numpy(np.random.randint(0, y1.shape[0], 32)).long()
        bs = y1.shape[0]
        converge = (self.kl(y1, y2) + self.kl(y2, y1)).sum() / bs
        diverge = (-self.kl(y1, y3) - self.kl(y2, y3)).sum() / bs
        #import pdb; pdb.set_trace()
        self.spread = y1.sum(0) / y1.sum()
        entropy1 = -self.entropy(y1.sum(0) / y1.sum()) # maximize entropy of spread over all data points
        entropy2 = self.entropy(y1).mean() # minimize entropy of single data point

        #l4 = -self.kl(y3, y1)
        #l5 = self.entropy(y1) * 0.1
        #l6 = sum(dev) / len(dev)
        #import pdb; pdb.set_trace()
        loss = converge + diverge + entropy1 + entropy2 + 100
        #loss = l5

        #import pdb; pdb.set_trace()
        loss.backward()
        nn.utils.clip_grad_norm(self.encoder.parameters(), 5.0)
        self.optimizer.step()

        stats['Loss'] += loss.item()
        stats['Converge'] += converge.item()
        stats['Diverge'] += diverge.item()
        stats['Entropy1'] += entropy1.item()
        stats['Entropy2'] += entropy2.item()
        #stats['Dev'] += l6.item()

        return stats

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
    envs = [FourRoomsModEnv(gridsize=15, room_wh=(6, 6)),
            FourRoomsModEnv(gridsize=15, room_wh=(7, 7)),
            FourRoomsModEnv(gridsize=15, room_wh=(7, 7), close_doors=["west"])
            #FourRoomsModEnv(gridsize=15, room_wh=(6, 7)),
            ]
    a = AbstractMDPsContrastive(envs)
    a.train(max_epochs=200)
    #print(a.mean_t)
    print(a.spread)
    a.gen_plot()

