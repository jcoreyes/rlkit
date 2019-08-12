"""
Save abstract transitions
Score likelihoods just based on nondiagonal abstract transitions
"""

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
import matplotlib.pyplot as plt


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

    def compute_alpha(self, A, B, O, pi, eps=0):
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
        alpha /= (alpha.sum(-1, keepdims=True) + eps)
        return alpha

    def compute_beta(self, A, B, O, eps=0):
        num_seq = O.shape[0]
        beta = np.zeros((num_seq, 2, A.shape[0]))
        beta[:, 1, :] = 1
        beta[:, 0, :] = 0
        for i in range(A.shape[0]):
            for j in range(A.shape[0]):
                beta[:, 0, i] += A[i, j] * B[O[:, 1], j] * beta[:, 1, j]
        beta /= (beta.sum(-1, keepdims=True) + eps)
        return beta

    def zero_diag(self, A):
        keep = np.logical_not(np.eye(A.shape[0]))
        return A * keep

    def nondiag_trans(self, A):
        B = self.zero_diag(A)
        B /= ((B + 1e-8).sum(1))
        return B

    def get_nondiag_from_O(self, decoder, O):
        s1 = decoder[O[:, 0]].argmax(-1)
        s2 = decoder[O[:, 1]].argmax(-1)

        counts = np.zeros((self.abstract_dim, self.abstract_dim))
        for i in range(s1.shape[0]):
            counts[s1[i], s2[i]] += 1

        empirical_A = self.zero_diag(counts)
        empirical_A /= (empirical_A.sum(1, keepdims=True) + 1e-12)

        return empirical_A

    def compute_nondiag_trans(self, A, B, O, gamma):
        # compute probability that observations came from A only including nondiag transitions
        decoder = B / (B.sum(1, keepdims=True) + 1e-12)
        s1 = decoder[O[:, 0]].argmax(-1)
        s2 = decoder[O[:, 1]].argmax(-1)

        idx = s1 != s2

        counts = np.zeros((self.abstract_dim, self.abstract_dim))
        for i in range(s1.shape[0]):
            counts[s1[i], s2[i]] += 1


        #empirical_A = self.zero_diag(counts)
        #empirical_A /= (empirical_A.sum(1, keepdims=True) + 1e-12)

        a_dim = A.shape[0]
        pi = np.ones(a_dim) / a_dim
        A_nondiag = self.nondiag_trans(A)
        if idx.sum() > 0:
            O_new = O[idx, :]
            alpha = self.compute_alpha(A_nondiag, B, O_new, pi, eps=1e-20)
            beta = self.compute_beta(A_nondiag, B, O_new, eps=1e-20)
            gamma =  alpha * beta
        # counts = np.zeros((self.abstract_dim, self.abstract_dim))
        # for i in range(s1.shape[0]):
        #     counts[s1[i], s2[i]] += 1
        #
        # empirical_A = self.zero_diag(counts)
        # empirical_A /= (empirical_A.sum(1, keepdims=True) + 1e-12)
        #
        # model_A = self.zero_diag(A)
        # model_A /= (model_A.sum(1, keepdims=True) + 1e-12)
        #
        # kl = (model_A * (np.log(model_A + 1e-12) - np.log(empirical_A + 1e-12))).sum(-1).mean()
        # ll = np.exp(-kl)
            nondiag_ll = gamma
            nondiag_ll = nondiag_ll.sum(-1)[:, 0].mean()
        else:
            nondiag_ll = 0

        return nondiag_ll

    def compute_gamma_zeta(self, alpha, beta, A, B, O):

        gamma = alpha * beta
        likelihood = gamma.sum(-1)[:, 0].mean()

        # compute likelihood only based on non diag transitions

        # TODO Filter out rows of gamma that have the same hidden state as highest prob for both
        # time steps
        nondiag_ll = self.compute_nondiag_trans(A, B, O, gamma)
        # max_hidden = gamma.argmax(-1)
        # idx = max_hidden[:, 0] != max_hidden[:, 1]
        # if idx.sum() > 0:
        #     nondiag_ll = gamma[idx]
        #     nondiag_ll = nondiag_ll.sum(-1)[:, 0].mean()
        # else:
        #     nondiag_ll = 0
        #
        # print(idx.sum())

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
        return gamma, zeta, nondiag_ll, likelihood

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
                B = np.random.uniform(size=(n_states, self.abstract_dims[j]))
                B /= B.sum(0, keepdims=True)
                B_inner_lst.append(B)
            B_lst.append(B_inner_lst)
            O = np.array(self.envs[i].transitions)
            O_lst.append(O)

        mixture = np.zeros((n_envs, self.n_abstract_mdps))

        likelihood = np.zeros((n_envs, self.n_abstract_mdps))
        gamma_lst = [[[] for _ in range(n_envs)] for _ in range(self.n_abstract_mdps)]
        zeta_lst = [[[] for _ in range(n_envs)] for _ in range(self.n_abstract_mdps)]


        for epoch in range(1, max_epochs + 1):
            # E step
            for i in range(self.n_abstract_mdps):
                for j in range(n_envs):
                    alpha = self.compute_alpha(A_lst[i], B_lst[j][i], O_lst[j], pi_lst[i])
                    beta = self.compute_beta(A_lst[i], B_lst[j][i], O_lst[j])
                    gamma, zeta, nondiag_ll, ll = self.compute_gamma_zeta(alpha, beta, A_lst[i],
                                                                     B_lst[j][i], O_lst[j])
                    gamma_lst[i][j] = gamma
                    zeta_lst[i][j] = zeta
                    likelihood[j, i] = ll
                    mixture[j, i] = nondiag_ll + 1e-8

            mixture /= mixture.sum(-1, keepdims=True)
            #print(mixture)
            total_ll = (likelihood * mixture).sum() / self.n_envs

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

            for i in range(self.n_abstract_mdps):
                for j in range(self.n_envs):
                    B_lst[j][i] = self.compute_B(gamma_lst[i][j], O_lst[j], len(self.envs[j].states), self.abstract_dims[i])

        best_encoder = []
        best_A = []
        for i in range(self.n_envs):
            abstract_idx = mixture[i].argmax()
            best_A.append(abstract_idx)
            best_B = B_lst[i][abstract_idx]
            pi = pi_lst[abstract_idx].reshape((1, -1))
            decoder = best_B * pi# / best_B.sum(1, keepdims=True) + 1e-8

            decoder = decoder / (decoder.sum(1, keepdims=True) + 1e-12)


            best_encoder.append(decoder)

        total_ll = (likelihood * mixture).sum() / self.n_envs
        self.gen_plot(total_ll, try_no, best_encoder, mixture, best_A, A_lst, O_lst)
        # B /= B.sum(1, keepdims=True)
        # print(A)
        # print(B[:5])
        # self.encoder = B
        return total_ll, np.stack(A_lst), mixture

    def kl(self, dist1, dist2):
        return (dist1 * (torch.log(dist1 + 1e-8) - torch.log(dist2 + 1e-8))).sum(1)

    def entropy(self, dist):
        return -(dist * torch.log(dist + 1e-8)).sum(-1)


    def gen_plot(self, likelihood, i, encoder_lst, mixture, best_A, A_lst, O_lst):
        plots = [env.gen_plot(encoder_lst[j]) for j, env in enumerate(
            self.envs)]


        #fig = plt.figure()
        fig, axs = plt.subplots(1, len(plots), figsize=(10, 5))
        for j in range(len(plots)):
            #plots = np.concatenate(plots, 1)
            axs[j].imshow(plots[j])
            A = self.nondiag_trans(A_lst[best_A[j]])
            #A = self.get_nondiag_from_O(encoder_lst[j], O_lst[j])

            axs[j].set_title(np.array2string(A, formatter={'float_kind':lambda x: "%.2f" % x}))
            axs[j].set_xlabel(np.array2string(mixture[j], formatter={'float_kind':lambda x: "%.3f" %
                                                                                        x}))
        # mixture[j], A_lst[best_A[j]


        #plt.imshow(plots)
        plt.savefig('/home/jcoreyes/abstract/rlkit/examples/abstractmdp/exps/exp3/fig_%.3f_%d.png'
                    '' % (likelihood, i))
        #plt.show()
        plt.clf()


if __name__ == '__main__':
    # laplacian = Laplacian(FourRoomsModEnv())
    # vals, vectors= laplacian.generate_laplacian()
    # laplacian.gen_plot(vectors[:, 1])
    envs = [FourRoomsModEnv(gridsize=11, room_wh=(5, 5)),
            FourRoomsModEnv(gridsize=11, room_wh=(5, 4)),
            FourRoomsModEnv(gridsize=11, room_wh=(5, 5), close_doors=["north", "south"]),
            FourRoomsModEnv(gridsize=11, room_wh=(5, 4), close_doors=["north", "south"])
            #FourRoomsModEnv(gridsize=15, room_wh=(7, 6)),
            #TwoRoomsModEnv(gridsize=11, room_w=5),
            #TwoRoomsModEnv(gridsize=11, room_w=5, door_pos=1)
            #TwoRoomsModEnv(gridsize=15, room_w=6)
            #FourRoomsModEnv(gridsize=15, room_wh=(7, 7), close_doors=["west"])
            #FourRoomsModEnv(gridsize=15, room_wh=(6, 7)),
            ]
    tries = 1500


    data = [[], [], []]
    for i in range(tries):
        a = AbstractMDPsContrastive(envs)
        likelihood, A, mixture = a.train(i, max_epochs=300) # 300
        print(likelihood)
        print(A)
        print(mixture)
        data[0].append(A)
        data[1].append(mixture)
        data[2].append(likelihood)
        #print(a.mean_t)
        #print(a.y1)
        #a.gen_plot(likelihood, i)
    save_dir = '/home/jcoreyes/abstract/rlkit/examples/abstractmdp/exps/exp2/'
    np.save(save_dir + 'abstract_t.npy', np.stack(data[0]))
    np.save(save_dir + 'mixture.npy', np.stack(data[1]))
    np.save(save_dir + 'likelihood.npy', np.array(data[2]))


