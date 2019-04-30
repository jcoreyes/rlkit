"""
Separate self transitions vs other transitions.
"""
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

    def true_values(self, s, option=0):
        x = s[:, 0]
        y = s[:, 1]
        x1 = x // 7
        x2 = y // 7

        z = np.zeros((x.shape[0], 4))

        indices = [np.bitwise_and(x1, x2),
                   np.bitwise_and(x1 == 1, x2 == 0),
                   np.bitwise_and(x1 == 0, x2 == 0),
                   np.bitwise_and(x1 == 0, x2 == 1)]
        if option == 'optimal':
            z[np.bitwise_and(x1, x2), 0] = 1
            z[np.bitwise_and(x1 == 1, x2 == 0), 1] = 1
            z[np.bitwise_and(x1 == 0, x2 == 0), 2] = 1
            z[np.bitwise_and(x1 == 0, x2 == 1), 3] = 1

        if option == 1:
            z[np.bitwise_and(x1, x2), 1] = 0.51
            z[np.bitwise_and(x1, x2), 3] = 0.49

            z[np.bitwise_and(x1 == 1, x2 == 0), 1] = 0.51
            z[np.bitwise_and(x1 == 1, x2 == 0), 3] = 0.49

            z[np.bitwise_and(x1==0, x2==0), 0] = 0.51
            z[np.bitwise_and(x1 == 0, x2 == 0), 2] = 0.49

            z[np.bitwise_and(x1 == 0, x2 == 1), 0] = 0.51
            z[np.bitwise_and(x1 == 0, x2 == 1), 2] = 0.49

        if option == 'uniform':
            z[:] = 0.25

        if option == 'optimal_uniform':
            for i, idx in enumerate(indices):
                z[idx, i] = 0.28
                for j in range(4):
                    if i != j:
                        z[idx, j] = 0.24

        if option == 'onestate':
            z[:, 0] = 1

        if option == 'onestate_uniiform':
            z[:] = 0.24
            z[:, 0] = 0.28
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

        self.n_envs = len(self.envs)
        self.n_abstract_mdps = 2
        self.abstract_dim = 4
        self.state_dim = 4
        self.states = []
        self.state_to_idx = None

        all_encoder_lst = nn.ModuleList()
        for i in range(self.n_envs):
            encoder_lst = nn.ModuleList()
            for j in range(self.n_abstract_mdps):
                encoder = Mlp((128, 128, 128), output_size=self.abstract_dim, input_size=self.state_dim,
                           output_activation=F.softmax, layer_norm=True)
                encoder.apply(init_weights)
                encoder_lst.append(encoder)

            all_encoder_lst.append(encoder_lst)
        self.all_encoder_lst = all_encoder_lst

        self.optimizer = optim.Adam(self.all_encoder_lst.parameters(), lr=1e-4)

    def train(self, max_epochs=100):
        a_dim = self.abstract_dim

        mixture = from_numpy(np.ones((self.n_envs, self.n_abstract_mdps)) / self.n_abstract_mdps)
        all_abstract_t = from_numpy(np.random.uniform(size=(self.n_abstract_mdps, a_dim, a_dim)))
        self_t = from_numpy(np.zeros((self.n_envs, a_dim)) + 0.9)
        all_abstract_t[:, np.arange(a_dim), np.arange(a_dim)] = 0
        all_abstract_t /= (all_abstract_t.sum(1, keepdim=True) + 1e-8)
        for epoch in range(1, max_epochs + 1):
            stats, abstract_t = self.train_epoch(epoch, mixture, all_abstract_t, self_t)
            print(stats)
            print(mixture)
            print(abstract_t)

        self.gen_plot(mixture)

    def kl(self, dist1, dist2):
        return (dist1 * (torch.log(dist1 + 1e-8) - torch.log(dist2 + 1e-8))).sum(1)

    def entropy(self, dist):
        return -(dist * torch.log(dist + 1e-8)).sum(-1)



    def compute_empirical_t(self, env, encoder, hardcounts=False):
        trans = env.transitions_np
        s1 = trans[:, :4]
        s2 = trans[:, 4:]

        y1 = encoder(from_numpy(s1))
        y2 = encoder(from_numpy(s2))

        a_t = from_numpy(np.zeros((self.abstract_dim, self.abstract_dim)))
        for i in range(self.abstract_dim):
            for j in range(self.abstract_dim):
                if i == j:
                    continue
                if hardcounts:
                    a_t[i, j] += ((y1.max(-1)[1] == i).float() * (y2.max(-1)[1] == j).float()).sum()
                else:
                    a_t[i, j] += (y1[:, i] * y2[:, j]).sum(0)

        n_a_t = from_numpy(np.zeros((self.abstract_dim, self.abstract_dim)))
        for i in range(self.abstract_dim):
            n_a_t[i, :] += a_t[i, :] / (a_t[i, :].sum() + 1e-8)


        return n_a_t

    def encode_transitions(self, env, encoder):
        trans = env.transitions_np
        s1 = trans[:, :4]
        s2 = trans[:, 4:]
        y1 = encoder(from_numpy(s1))
        y2 = encoder(from_numpy(s2))
        y3 = encoder(from_numpy(env.sample_states(s1.shape[0])))
        return y1, y2, y3


    def train_epoch(self, epoch, mixture, all_abstract_t, self_t):


        # train encoder
        for i in range(5):
            stats, all_concrete_ll = self.train_encoder(epoch, mixture, all_abstract_t, self_t)

        # compute mixture components
        mixture[:] = all_concrete_ll / (all_concrete_ll.sum(1, keepdim=True) + 1e-12)
        #import pdb; pdb.set_trace()
        # compute abstract transitions using mixture
        all_empirical_t = []
        for i in range(self.n_abstract_mdps):
            empirical_t = [self.compute_empirical_t(self.envs[j], self.all_encoder_lst[j][i], hardcounts=False) for j in range(self.n_envs)]
            all_empirical_t.append(empirical_t)

        for i in range(self.n_abstract_mdps):
            all_abstract_t[i] = (mixture[:, i].view(-1, 1, 1) * torch.stack(all_empirical_t[i])).sum(0)

        all_abstract_t[:, np.arange(self.abstract_dim), np.arange(self.abstract_dim)] = 0
        all_abstract_t[:] = all_abstract_t / (all_abstract_t.sum(-1, keepdim=True) + 1e-12)

        # Compute self transitions
        for i in range(self.n_envs):
            self_t[i, :] = all_empirical_t[mixture[i, :].argmax()][i][np.arange(self.abstract_dim), np.arange(self.abstract_dim)]


        return stats, all_abstract_t

    def compute_abstract_trans_ll(self, y1, y2, A, self_t):
        # Compute likelihood only using non diagonal
        A_nondiag = A.clone()
        A_nondiag[np.arange(A.shape[0]), np.arange(A.shape[0])] = 0
        A_nondiag /= (A_nondiag.sum(1, keepdim=True) + 1e-8)

        a_ll = from_numpy(np.zeros(1))
        loss_ll = from_numpy(np.zeros(1))
        for i in range(self.abstract_dim):
            for j in range(self.abstract_dim):
                if i == j:
                    continue
                    loss_ll += (self_t[i].detach() * y1[:, i] * y2[:, j] * torch.log(self_t[i].detach() + 1e-12)).sum()
                else:
                    #((1-self_t[i].detach())
                    val = (y1[:, i] * y2[:, j] * torch.log(A_nondiag[i, j].detach() + 1e-12)).sum()
                    a_ll += val # TODO 1 - self_t here?
                    loss_ll += val
        #a_ll /= (y1.shape[0] * self.abstract_dim)
        #import pdb; pdb.set_trace()
        return a_ll, loss_ll


    def train_encoder(self, epoch, mixture, all_abstract_t, self_t):
        stats = OrderedDict([('Loss', 0), ('Entropy1', 0), ('Entropy2', 0),])

        #compute likelihood under each abstract mdp weighted by mixture
        self.optimizer.zero_grad()
        all_concrete_ll = from_numpy(np.zeros(mixture.shape))
        loss = from_numpy(np.zeros(1))
        for i in range(self.n_envs):
            for j in range(self.n_abstract_mdps):
                y1, y2, y3 = self.encode_transitions(self.envs[i], self.all_encoder_lst[i][j])
                concrete_ll, loss_ll = self.compute_abstract_trans_ll(y1, y2, all_abstract_t[j], self_t[i])
                all_concrete_ll[i, j] = concrete_ll.detach()
                marginal_entropy = self.entropy(y3.sum(0) / y3.sum())
                loss += mixture[i, j] * (loss_ll + 1000*marginal_entropy)

                stats['Entropy1'] += marginal_entropy.item()

        loss *= -1
        loss.backward()
        nn.utils.clip_grad_norm(self.all_encoder_lst.parameters(), 5.0)
        self.optimizer.step()

        stats['Loss'] += loss.item()

        return stats, all_concrete_ll

    def gen_plot(self, mixture):
        plots = []

        for i in range(self.n_envs):
            best_encoder = self.all_encoder_lst[i][mixture[i].argmax()]
            plots.append(self.envs[i].gen_plot(best_encoder))

        plots = np.concatenate(plots, 1)

        plt.imshow(plots)
        #plt.savefig('/home/jcoreyes/abstract/rlkit/examples/abstractmdp/fig1.png')
        plt.show()


if __name__ == '__main__':
    # laplacian = Laplacian(FourRoomsModEnv())
    # vals, vectors= laplacian.generate_laplacian()
    # laplacian.gen_plot(vectors[:, 1])
    envs = [FourRoomsModEnv(gridsize=15, room_wh=(7, 7)),
            #FourRoomsModEnv(gridsize=15, room_wh=(7, 7)),
            #FourRoomsModEnv(gridsize=15, room_wh=(6, 6)),
            FourRoomsModEnv(gridsize=15, room_wh=(7, 7), close_doors=["west"]),
            #FourRoomsModEnv(gridsize=15, room_wh=(7, 7), close_doors=["west"])
            #FourRoomsModEnv(gridsize=15, room_wh=(6, 7)),
            ]
    a = AbstractMDPsContrastive(envs)
    a.train(max_epochs=100)
    #print(a.mean_t)
    #print(a.y1)
    #print(a.a_t)
    #a.gen_plot()

