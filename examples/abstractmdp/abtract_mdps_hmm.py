import numpy as np


import gym
import numpy as np
import gym_minigrid
from gym_minigrid.envs.fourrooms import FourRoomsModEnv, BridgeEnv, WallEnv, TwoRoomsModEnv
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

class AbstractMDPsContrastive:
    def __init__(self, envs):
        self.envs = [EnvContainer(env) for env in envs]

        self.n_abstract_mdps = 2
        self.abstract_dim = 3
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
        A = np.random.uniform(size=(self.abstract_dim, self.abstract_dim)) # P(q_t+1=i | q_t=j)
        # A = np.ones((self.abstract_dim, self.abstract_dim))
        for i in range(self.abstract_dim):
             A[i, i] += 10
        A /= A.sum(1, keepdims=True)

        B = np.ones((n_states, self.abstract_dim)) / n_states # P(o_t|q_t=i)
        #B = np.random.uniform(size=(n_states, self.abstract_dim)) + 100
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

            likelihood = gamma.sum(-1)[:, 0].mean()
            print(likelihood)
            # normalize gamma
            gamma = gamma / gamma.sum(-1, keepdims=True) # P(q_t=j|O, lambda)



            zeta = np.zeros((num_seq, self.abstract_dim, self.abstract_dim)) # P(q_t=i, q_t+1=j|O, lambda)
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
            if likelihood == prev_likelihood and likelihood > 0.5:
                break
            prev_likelihood = likelihood

            #print(B[:5])
        #import pdb; pdb.set_trace()

        B /= B.sum(1, keepdims=True)
        print(A)
        print(B[:5])
        self.encoder = B
        return likelihood

    def kl(self, dist1, dist2):
        return (dist1 * (torch.log(dist1 + 1e-8) - torch.log(dist2 + 1e-8))).sum(1)

    def entropy(self, dist):
        return -(dist * torch.log(dist + 1e-8)).sum(-1)


    def gen_plot(self, likelihood, i):
        plots = [env.gen_plot(self.encoder) for env in self.envs]

        plots = np.concatenate(plots, 1)

        plt.imshow(plots)
        plt.savefig('/home/jcoreyes/abstract/rlkit/examples/abstractmdp/exps/exp3/fig_%.6f_%d.png' % (likelihood, i))
        #plt.show()


if __name__ == '__main__':
    # laplacian = Laplacian(FourRoomsModEnv())
    # vals, vectors= laplacian.generate_laplacian()
    # laplacian.gen_plot(vectors[:, 1])
    envs = [#FourRoomsModEnv(gridsize=15, room_wh=(6, 6)),
            #FourRoomsModEnv(gridsize=15, room_wh=(7, 7)),
            #TwoRoomsModEnv(gridsize=15, room_w=7)
            BridgeEnv(),
            #FourRoomsModEnv(gridsize=15, room_wh=(7, 7), close_doors=["west"])
            #FourRoomsModEnv(gridsize=15, room_wh=(6, 7)),
            ]
    tries = 100

    for i in range(tries):
        a = AbstractMDPsContrastive(envs)
        likelihood = a.train(max_epochs=300)
        #print(a.mean_t)
        #print(a.y1)
        a.gen_plot(likelihood, i)

