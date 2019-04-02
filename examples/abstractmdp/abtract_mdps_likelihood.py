import numpy as np


import gym
import numpy as np
import gym_minigrid
from gym_minigrid.envs.fourrooms import FourRoomsModEnv, BridgeEnv, WallEnv
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from rlkit.torch.networks import Mlp
from rlkit.torch.pytorch_util import from_numpy, get_numpy, init_weights
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

    def true_values(self, s):
        x = s[:, 0]
        y = s[:, 1]
        x1 = x // 7
        x2 = y // 7

        z = np.zeros((x.shape[0], 4))

        z[np.bitwise_and(x1, x2), 1] = 0.51
        z[np.bitwise_and(x1, x2), 3] = 0.49

        z[np.bitwise_and(x1 == 1, x2 == 0), 1] = 0.51
        z[np.bitwise_and(x1 == 1, x2 == 0), 3] = 0.49

        z[np.bitwise_and(x1==0, x2==0), 0] = 0.51
        z[np.bitwise_and(x1 == 0, x2 == 0), 2] = 0.49

        z[np.bitwise_and(x1 == 0, x2 == 1), 0] = 0.51
        z[np.bitwise_and(x1 == 0, x2 == 1), 2] = 0.49




        #z[:] = 0.25

        return from_numpy(z)

    def gen_plot(self, encoder):

        #X = np.arange(0, self.width)
        #Y = np.arange(0, self.height)
        #X, Y = np.meshgrid(X, Y)
        Z_lst = []
        # for i in range(4):
        #     Z = np.zeros((self.width, self.height))
        #     for state in self.states:
        #         dist = get_numpy(encoder(from_numpy(np.array(state)).unsqueeze(0)))
        #         #dist = get_numpy(self.true_values(np.array(state).reshape((1, -1))))
        #         Z[state[:2]] = dist[0, i]
        #     Z_lst.append(Z)
        Z = np.zeros((self.width, self.height))
        for state in self.states:
            dist = get_numpy(encoder(from_numpy(np.array(state)).unsqueeze(0)))
            # dist = get_numpy(self.true_values(np.array(state).reshape((1, -1))))
            Z[state[:2]] = np.argmax(dist) + 1
        Z_lst.append(Z)

        return np.concatenate(Z_lst, 0)

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

        self.encoder = Mlp((128, 128, 128), output_size=self.abstract_dim, input_size=self.state_dim,
                           output_activation=F.softmax, layer_norm=True)

        self.encoder.apply(init_weights)
        self.transitions = nn.Parameter(torch.zeros((self.abstract_dim, self.abstract_dim)))

        self.optimizer = optim.Adam(self.encoder.parameters(), lr=1e-4)

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
            if stats['Loss'] < 221.12:
               break
            print(stats)

    def kl(self, dist1, dist2):
        return (dist1 * (torch.log(dist1 + 1e-8) - torch.log(dist2 + 1e-8))).sum(1)

    def entropy(self, dist):
        return -(dist * torch.log(dist + 1e-8)).sum(-1)



    def compute_abstract_t(self, env):
        trans = env.transitions_np
        s1 = trans[:, :4]
        s2 = trans[:, 4:]
        all_states = self.encoder(from_numpy(env.all_states()))
        y1 = self.encoder(from_numpy(s1))
        y2 = self.encoder(from_numpy(s2))
        y3 = self.encoder(from_numpy(env.sample_states(s1.shape[0])))

        # Hardcode if y1 and y2 were what you wanted
        #y1 = env.true_values(s1)
        #y2 = env.true_values(s2)


        a_t = from_numpy(np.zeros((self.abstract_dim, self.abstract_dim)))
        for i in range(self.abstract_dim):
            for j in range(self.abstract_dim):
                a_t[i, j] += (y1[:, i] * y2[:, j]).sum(0)

        n_a_t = from_numpy(np.zeros((self.abstract_dim, self.abstract_dim)))
        for i in range(self.abstract_dim):
            n_a_t[i, :] += a_t[i, :] / (a_t[i, :].sum() + 1e-8)


        return n_a_t, y1, y2, y3, all_states


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
        y1 = torch.cat([x[4] for x in data], 0)


        a_t_loss = - sum([torch.log(x  * from_numpy(np.eye(4))+ 1e-8).sum() for x in abstract_t]) / len(abstract_t)
        #a_t_loss = - sum([torch.log(x + 1e-8).sum() for x in abstract_t])
        entropy1 = -self.entropy(y1.sum(0) / y1.sum())   # maximize entropy of spread over all data points
        loss = a_t_loss #+ entropy1

        #self.y1 = y1
        #self.a_t = abstract_t[0]

        loss.backward()
        nn.utils.clip_grad_norm(self.encoder.parameters(), 5.0)
        self.optimizer.step()

        stats['Loss'] += loss.item()
        #stats['Converge'] += converge.item()
        #stats['Diverge'] += diverge.item()
        stats['Entropy1'] += entropy1.item()
        #stats['Entropy2'] += entropy2.item()
        #stats['       Dev'] += l6.item()

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
    envs = [FourRoomsModEnv(gridsize=15, room_wh=(7, 7)),
            FourRoomsModEnv(gridsize=15, room_wh=(6, 6)),
            #FourRoomsModEnv(gridsize=15, room_wh=(7, 7), close_doors=["west"])
            FourRoomsModEnv(gridsize=15, room_wh=(6, 7)),
            ]
    a = AbstractMDPsContrastive(envs)
    a.train(max_epochs=300)
    #print(a.mean_t)
    #print(a.y1)
    #print(a.a_t)
    a.gen_plot()

