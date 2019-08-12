"""
Evaluate likelihood with true encodings
"""

import gym
import numpy as np
#import gym_minigrid
from rlkit.envs.gym_minigrid.envs.fourrooms import FourRoomsModEnv, BridgeEnv, WallEnv, TwoRoomsModEnv
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

    def get_decoding(self):
        s = self.states_np
        x = s[:, 0]
        y = s[:, 1]

        z = np.zeros((x.shape[0], 4))

        def get_class_idx(x, y):
            mid_x = 5
            mid_y = 5
            if x < mid_x and y < mid_y:
                return 0
            elif x >= mid_x and y < mid_y:
                return 1
            elif x < mid_x and y >= mid_y:
                return 2
            else:
                return 3

        for i, s in enumerate(self.states):
            z[i, get_class_idx(*s)] = 1

        return z


class AbstractMDPsContrastive:
    def __init__(self, envs):
        self.envs = [EnvContainer(env) for env in envs]
        self.n_envs = len(self.envs)
        self.n_abstract_mdps = 2
        self.abstract_dim = 4
        self.abstract_dims = [4, 4]
        self.state_dim = 4
        self.states = []
        self.state_to_idx = None

        # self.encoder = Mlp((64, 64, 64), output_size=self.abstract_dim, input_size=self.state_dim,
        #                    output_activation=F.softmax, layer_norm=True)
        # self.transitions = nn.Parameter(torch.zeros((self.abstract_dim, self.abstract_dim)))
        #
        # self.optimizer = optim.Adam(self.encoder.parameters())

    def compute_alpha(self, A, B, O, pi):
        # print(A.shape)
        # print(B.shape)
        # print(O.shape)
        num_seq = O.shape[0]
        alpha = np.zeros((num_seq, 2, A.shape[0]))

        alpha[:, 0, :] = pi * B[O[:, 0], :]
        alpha[:, 1, :] = 0
        for j in range(A.shape[0]):
            for i in range(A.shape[0]):
                alpha[:, 1, j] += alpha[:, 0, i] * A[i, j] * B[O[:, 1], j]
        alpha /= alpha.sum(-1, keepdims=True)
        return alpha

    def compute_beta(self, A, B, O):
        num_seq = O.shape[0]
        beta = np.zeros((num_seq, 2, A.shape[0]))
        beta[:, 1, :] = 1
        beta[:, 0, :] = 0
        for i in range(A.shape[0]):
            for j in range(A.shape[0]):
                beta[:, 0, i] += A[i, j] * B[O[:, 1], j] * beta[:, 1, j]
        beta /= beta.sum(-1, keepdims=True)
        return beta

    def compute_gamma_zeta(self, alpha, beta, A, B, O):

        gamma = alpha * beta
        likelihood = gamma.sum(-1)[:, 0].mean()
        num_seq = alpha.shape[0]
        # normalize gamma
        gamma = gamma / gamma.sum(-1, keepdims=True)  # P(q_t=j|O, lambda)
        #pi = gamma[:, 0, :].mean(0)
        #import pdb; pdb.set_trace()
        zeta = np.zeros((num_seq, A.shape[0], A.shape[0]))  # P(q_t=i, q_t+1=j|O, lambda)
        norm = (alpha[:, 0, :] * beta[:, 0, :]).sum(-1)
        for i in range(A.shape[0]):
            for j in range(A.shape[0]):
                zeta[:, i, j] = alpha[:, 0, i] * A[i, j] * B[O[:, 1], j] * beta[:, 1, j] / norm
        return gamma, zeta, likelihood

    def compute_A(self, zeta, a_dim):
        A = np.zeros((a_dim, a_dim))
        for i in range(a_dim):
            for j in range(a_dim):
                A[i, j] = zeta[:, i, j].sum() / zeta[:, i, :].sum()
        return A

    def compute_B(self, gamma, O, n_states, a_dim):
        B = np.zeros((n_states, a_dim))
        for k in range(n_states):
            mask = O == k
            mask = np.expand_dims(mask, -1)
            B[k, :] = (gamma * mask).sum(1).sum(0) / gamma.sum(1).sum(0)
        return B

    def train(self, try_no, max_epochs=100):

        # Initialize A and B
        A_lst = []
        pi_lst = []
        for i in range(self.n_abstract_mdps):
            a_dim = self.abstract_dims[i]
            A = np.random.uniform(size=(a_dim, a_dim)) # P(q_t+1=i | q_t=j)
            pi = np.ones(a_dim)
            # A = np.ones((self.abstract_dim, self.abstract_dim))
            for j in range(a_dim):
               A[j, j] += 10

            #if i == 0:
            #    pi[:2] = 0.3
                #A[:, :2] = 0.1

            A /= A.sum(-1, keepdims=True)
            A_lst.append(A)


            pi /= pi.sum()
            pi_lst.append(pi)

        B_lst = []
        O_lst = []
        n_envs = len(self.envs)
        for i, env in enumerate(self.envs):
            n_states = len(env.states)
            B_inner_lst = []
            for j in range(self.n_abstract_mdps):
                #B = np.ones((n_states, self.abstract_dims[j])) / n_states # P(o_t|q_t=i)
                #B = np.random.uniform(size=(n_states, self.abstract_dims[j]))
                B = env.get_decoding()
                B /= B.sum(0, keepdims=True)
                B_inner_lst.append(B)
            B_lst.append(B_inner_lst)
            O = np.array(self.envs[i].transitions)
            O_lst.append(O)

        #mixture = np.zeros((n_envs, self.n_abstract_mdps))
        mixture = np.array([[0.9, 0.1],
                           [0.1, 0.9]])

        likelihood = np.zeros((n_envs, self.n_abstract_mdps))
        gamma_lst = [[[] for _ in range(n_envs)] for _ in range(self.n_abstract_mdps)]
        zeta_lst = [[[] for _ in range(n_envs)] for _ in range(self.n_abstract_mdps)]


        for epoch in range(1, max_epochs + 1):
            # E step
            for i in range(self.n_abstract_mdps):
                for j in range(n_envs):
                    alpha = self.compute_alpha(A_lst[i], B_lst[j][i], O_lst[j], pi_lst[i])
                    beta = self.compute_beta(A_lst[i], B_lst[j][i], O_lst[j])
                    gamma, zeta, ll = self.compute_gamma_zeta(alpha, beta, A_lst[i], B_lst[j][i], O_lst[j])
                    gamma_lst[i][j] = gamma
                    zeta_lst[i][j] = zeta
                    likelihood[j, i] = ll
                    #mixture[j, i] = ll
            #mixture /= mixture.sum(-1, keepdims=True)
            total_ll = (likelihood * mixture).sum() / self.n_envs
            #print(mixture)
            #print(total_ll)
            # M Step
            for i in range(self.n_abstract_mdps):
                #A = A_lst[i]
                A_lst[i][:] = 0
                for j in range(n_envs):
                    A = self.compute_A(zeta_lst[i][j], self.abstract_dims[i])
                    A /= A.sum(1, keepdims=True)
                    A_lst[i] += mixture[j, i] * A
                A_lst[i] = A_lst[i] / A_lst[i].sum(1, keepdims=True)

                    # normalize A

            #for i in range(self.n_abstract_mdps):
            #    for j in range(self.n_envs):
            #        B_lst[j][i] = self.compute_B(gamma_lst[i][j], O_lst[j], len(self.envs[j].states), self.abstract_dims[i])

        best_encoder = []
        for i in range(self.n_envs):
            abstract_idx = mixture[i].argmax()
            best_B = B_lst[i][abstract_idx]
            pi = pi_lst[abstract_idx].reshape((1, -1))
            decoder = best_B * pi# / best_B.sum(1, keepdims=True) + 1e-8

            decoder = decoder / (decoder.sum(1, keepdims=True) + 1e-12)


            best_encoder.append(decoder)

        total_ll = (likelihood * mixture).sum() / self.n_envs
        self.gen_plot(total_ll, try_no, best_encoder)
        # B /= B.sum(1, keepdims=True)
        # print(A)
        # print(B[:5])
        # self.encoder = B
        return total_ll, np.stack(A_lst), mixture

    def kl(self, dist1, dist2):
        return (dist1 * (torch.log(dist1 + 1e-8) - torch.log(dist2 + 1e-8))).sum(1)

    def entropy(self, dist):
        return -(dist * torch.log(dist + 1e-8)).sum(-1)


    def gen_plot(self, likelihood, i, encoder_lst):
        plots = [env.gen_plot(encoder_lst[j]) for j, env in enumerate(self.envs)]

        plots = np.concatenate(plots, 1)

        plt.imshow(plots)
        plt.savefig('/home/jcoreyes/abstract/rlkit/examples/abstractmdp/exps/true_decoding/fig_%.3f_%d.png' % (likelihood, i))
        #plt.show()


if __name__ == '__main__':
    # laplacian = Laplacian(FourRoomsModEnv())
    # vals, vectors= laplacian.generate_laplacian()
    # laplacian.gen_plot(vectors[:, 1])
    envs = [FourRoomsModEnv(gridsize=11, room_wh=(5, 5)),
            #FourRoomsModEnv(gridsize=11, room_wh=(5, 4)),
            FourRoomsModEnv(gridsize=11, room_wh=(5, 5), close_doors=["north", "south"]),
            #FourRoomsModEnv(gridsize=11, room_wh=(5, 4), close_doors=["north", "south"])
            #FourRoomsModEnv(gridsize=15, room_wh=(7, 6)),
            #TwoRoomsModEnv(gridsize=11, room_w=5),
            #TwoRoomsModEnv(gridsize=11, room_w=5, door_pos=1)
            #TwoRoomsModEnv(gridsize=15, room_w=6)
            #FourRoomsModEnv(gridsize=15, room_wh=(7, 7), close_doors=["west"])
            #FourRoomsModEnv(gridsize=15, room_wh=(6, 7)),
            ]
    tries = 1


    data = [[], [], []]
    for i in range(tries):
        a = AbstractMDPsContrastive(envs)
        likelihood, A, mixture = a.train(i, max_epochs=300)
        print(likelihood)
        print(A)
        print(mixture)
        data[0].append(A)
        data[1].append(mixture)
        data[2].append(likelihood)
        #print(a.mean_t)
        #print(a.y1)
        #a.gen_plot(likelihood, i)
    save_dir = '/home/jcoreyes/abstract/rlkit/examples/abstractmdp/exps/true_decoding/'
    np.save(save_dir + 'abstract_t.npy', np.stack(data[0]))
    np.save(save_dir + 'mixture.npy', np.stack(data[1]))
    np.save(save_dir + 'likelihood.npy', np.array(data[2]))


