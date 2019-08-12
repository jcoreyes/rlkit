"""
Evalulate likelihood on nondiagonal transitions
"""

import gym
import numpy as np
#import gym_minigrid
from rlkit.envs.gym_minigrid.envs.fourrooms import FourRoomsModEnv, BridgeEnv, WallEnv, TwoRoomsModEnv
from rlkit.envs.gym_minigrid.modenvs import RandomModEnv
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
import numpy as np


class EnvContainer:
    def __init__(self, env):
        self.env = env
        self.width = self.env.grid.height
        self.height = self.env.grid.height
        #self.room_wh = self.env.room_wh

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
        next_states = [np.array(state)]
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

class AbstractMDPLikelihood:
    def __init__(self, envs):
        self.envs = [EnvContainer(env) for env in envs]

        self.n_abstract_mdps = 2
        self.abstract_dim = 4
        self.state_dim = 4
        self.states = []
        self.state_to_idx = None

        self.nondiag_ll = False


    def trans_prob(self, A, i, j):
        if not self.nondiag_ll:
            return A[i, j]
        else:
            if i == j:
                return A[i, j]
            else:
                B = A[i]
                B[i] = 0
                B /= (B.sum() + 1e-8)
                B *= (1 - A[i, j])
                return B[j]

    def train(self, A, max_epochs=100):

        # Initialize A and B
        n_states = len(self.envs[0].states)

        #B = np.ones((n_states, self.abstract_dim)) / n_states # P(o_t|q_t=i)
        B = np.random.uniform(size=(n_states, self.abstract_dim)) + 100
        B /= B.sum(0, keepdims=True)
        O = np.array(self.envs[0].transitions)

        num_seq = O.shape[0]

        alpha = np.zeros((num_seq, 2, self.abstract_dim)) # P(o_1..o_t, q_t=i | lambda)
        beta = np.zeros((num_seq, 2, self.abstract_dim)) # P(o_t+1...o_T | q_t=i, lambda)
        pi = np.ones(self.abstract_dim) / self.abstract_dim
        prev_likelihood = 0

        for epoch in range(1, max_epochs + 1):
            # E step
            alpha[:, 0, :] = pi * B[O[:, 0], :]
            alpha[:, 1, :] = 0
            for j in range(self.abstract_dim):
                for i in range(self.abstract_dim):
                    alpha[:, 1, j] += alpha[:, 0, i] * self.trans_prob(A, i, j) * B[O[:, 1], j]

            beta[:, 1, :] = 1
            beta[:, 0, :] = 0
            for i in range(self.abstract_dim):
                for j in range(self.abstract_dim):
                    beta[:, 0, i] += self.trans_prob(A, i, j) * B[O[:, 1], j] * beta[:, 1, j]
            #import pdb; pdb.set_trace()
            # Normalize to deal with underflow
            alpha /= alpha.sum(-1, keepdims=True)
            beta /= beta.sum(-1, keepdims=True)

            gamma = alpha * beta

            likelihood = gamma.sum(-1)[:, 0].mean()
            #print(likelihood)
            # normalize gamma
            gamma = gamma / gamma.sum(-1, keepdims=True) # P(q_t=j|O, lambda)

            zeta = np.zeros((num_seq, self.abstract_dim, self.abstract_dim)) # P(q_t=i, q_t+1=j|O, lambda)
            norm = (alpha[:, 0, :] * beta[:, 0, :]).sum(-1)
            for i in range(self.abstract_dim):
                for j in range(self.abstract_dim):
                    zeta[:, i, j] = alpha[:, 0, i] * self.trans_prob(A, i, j) * B[O[:, 1], j] * beta[:, 1, j] / norm

            # M Steps
            for k in range(n_states):
                mask = O == k
                mask = np.expand_dims(mask, -1)
                B[k, :] = (gamma * mask).sum(1).sum(0) / gamma.sum(1).sum(0)

            # if likelihood == prev_likelihood and likelihood > 0.5:
            #     break
            prev_likelihood = likelihood

        B /= B.sum(1, keepdims=True)
        #print(A)
        #print(B[:5])
        self.encoder = B
        return likelihood, B

    def kl(self, dist1, dist2):
        return (dist1 * (torch.log(dist1 + 1e-8) - torch.log(dist2 + 1e-8))).sum(1)

    def entropy(self, dist):
        return -(dist * torch.log(dist + 1e-8)).sum(-1)


    def gen_plot(self, likelihood, i, encoder_lst, envs):
        plots = [EnvContainer(env).gen_plot(encoder_lst[j]) for j, env in enumerate(envs)]

        plots = np.concatenate(plots, 1)

        plt.imshow(plots)
        plt.savefig('/home/jcoreyes/abstract/rlkit/examples/abstractmdp/exps/exp_random/fig_%.6f_%d.png' % (likelihood, i))
        #plt.show()

if __name__ == '__main__':
    envs = [FourRoomsModEnv(gridsize=11, room_wh=(5, 5)),
            #FourRoomsModEnv(gridsize=11, room_wh=(5, 4)),
            FourRoomsModEnv(gridsize=11, room_wh=(5, 5), close_doors=["north", "south"]),
            #FourRoomsModEnv(gridsize=11, room_wh=(5, 4), close_doors=["north", "south"]),
            RandomModEnv(gridsize=11)
            ]
    tries = 1


    save_dir = '/home/jcoreyes/abstract/rlkit/examples/abstractmdp/exps/exp2/'
    abstract_t_all = np.load(save_dir + 'abstract_t.npy') # (n_tries, n_abstract_mdps, abstract_dim, abstract_dim)
    mixture = np.load(save_dir + 'mixture.npy')
    likelihood = np.load(save_dir + 'likelihood.npy')


    #abstract_t = abstract_t_all[likelihood.argmax()]


    abstract_t = np.array([
        [
            [0.9729, 0.01427, 0.01282, 0],
            [0.0154, 0.97224, 0, 0.012295],
            [0.01389, 0, 0.97237, 0.01373],
            [0,       0.0132, 0.0148, 0.972]
        ],
        [
            [0.98413, 0.01443, 0.0014283, 0],
            [0.015623, 0.9830, 0, 0.001368],
            [0.00154, 0, 0.98408, 0.014379],
            [0, 0.001468, 0.01555, 0.98297]
        ]]
    )
    # abstract_t = np.array([
    #     [
    #         [0.95, 0.025, 0.025, 0],
    #         [0.025, 0.95, 0, 0.025],
    #         [0.025, 0, 0.95, 0.025],
    #         [0,       0.025, 0.025, 0.95]
    #     ],
    #     [
    #         [0.9, 0.1, 0.00, 0],
    #         [0.1, 0.9, 0, 0.0],
    #         [0.00, 0, 0.9, 0.1],
    #         [0, 0.00, 0.1, 0.9]
    #     ]]
    # )
    abstract_t = (abstract_t + .1)
    abstract_t /= abstract_t.sum(2, keepdims=True)
    print(abstract_t)
    n_abstract_mdps = abstract_t.shape[0]
    n_envs = len(envs)
    best_ll = np.zeros((len(envs), n_abstract_mdps))
    best_encoder = [[0]*n_abstract_mdps for _ in range(n_envs)]
    encoder_lst = []
    for i, env in enumerate(envs):
        for abstract_idx in range(n_abstract_mdps):
            for j in range(tries):
                a = AbstractMDPLikelihood([env])
                likelihood, encoder = a.train(abstract_t[abstract_idx], max_epochs=300)
                if likelihood > best_ll[i, abstract_idx]:
                    best_ll[i, abstract_idx] = likelihood
                    best_encoder[i][abstract_idx] = encoder

    for i in range(len(envs)):
        encoder_lst.append(best_encoder[i][best_ll[i].argmax()])
    print(abstract_t)
    print(best_ll)
    a.gen_plot(0, 0, encoder_lst, envs)

        #print(a.mean_t)
        #print(a.y1)
        #a.gen_plot(likelihood, i)



