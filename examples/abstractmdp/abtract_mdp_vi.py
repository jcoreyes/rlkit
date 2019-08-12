import numpy as np


import gym
import numpy as np
import gym_minigrid
from gym_minigrid.envs.fourrooms import FourRoomsModEnv, BridgeEnv, WallEnv
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from rlkit.torch.networks import Mlp
from rlkit.torch.pytorch_util import from_numpy, get_numpy, set_gpu_mode
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data

class AbstractMDPVI:
    def __init__(self, env):
        self.env = env
        self.width = self.env.grid.height
        self.height = self.env.grid.height
        self.abstract_dim = 4
        self.state_dim = 2
        self.states = []
        self.state_to_idx = None

        self.encoder = Mlp((64, 64, 64), output_size=self.abstract_dim, input_size=self.state_dim,
                           output_activation=F.softmax, layer_norm=False)

        states = []
        for j in range(self.env.grid.height):
            for i in range(self.env.grid.width):
                if self.env.grid.get(i, j) == None:
                    states.append((i, j))

        self.states = states
        self.states_np = np.array(states)
        self.state_to_idx = {s:i for i, s in enumerate(self.states)}

        self.next_states = []
        for i, state in enumerate(states):
            next_states = self._gen_transitions(state)
            self.next_states.append(next_states)

        self.next_states = np.array(self.next_states)

        self.encoder.cuda()
        self.optimizer = optim.Adam(self.encoder.parameters(), lr=1e-4)

    def _gen_transitions(self, state):

        actions = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        next_states = []
        for action in actions:
            ns = np.array(state) + action
            if ns[0] >= 0 and ns[1] >= 0 and ns[0] < self.width and ns[1] < self.height and \
                self.env.grid.get(*ns) == None:
                next_states.append(self.state_to_idx[tuple(ns.tolist())])
            else:
                next_states.append(-1)
        return next_states

    def train_vi(self):
        dataset = data.TensorDataset(from_numpy(np.arange(len(self.states))))
        dataloader = data.DataLoader(dataset, batch_size=32, shuffle=True)

        self.qvalues = from_numpy(np.zeros((len(self.states), len(self.states), 4)))
        self.values = from_numpy(np.zeros((len(self.states), len(self.states))))

        states = from_numpy(self.states_np)
        next_states = from_numpy(self.next_states).long()

        def eval_reward(s1, s2):
            match = (s1 == s2).float()
            return (1 - match) * -1 + match * 0


        # indexed as goal state, then state
        for goal_state in range(len(self.states)):
            for train_itr in range(20):
                for batch_idx, s1 in enumerate(dataloader):
                    s1 = s1[0].long()
                    s2 = next_states[s1]
                    for i in range(s2.shape[1]):
                        ns = s2[:, i]
                        valid_trans = (ns != -1).float()
                        update = eval_reward(ns, goal_state) + 1.0 * self.values[goal_state, ns]
                        #update[ns==goal_state] = 0
                        # Overwrite invalid actions with high negative reward so not chosen by max in value update
                        self.qvalues[goal_state, s1, i] = valid_trans * update + (1 - valid_trans) * - 1000


                    self.values[goal_state, s1] = self.qvalues[goal_state, s1, :].max(-1)[0]
                    self.values[goal_state, goal_state] = 0

            print(float(goal_state)/(len(self.states)))
        #import pdb; pdb.set_trace()
        self.values -= 1
        self.values[np.arange(len(self.states)), np.arange(len(self.states))] = 0
        np.save("/home/jcoreyes/abstract/rlkit/examples/abstractmdp/values.npy", get_numpy(self.values))



    def test_vi(self):
        Z = np.zeros((self.width, self.height))
        values = get_numpy(self.values)
        for i, state in enumerate(self.states):
            Z[state] = values[0, i]
        print(Z)

        Z += Z.min()
        Z /= Z.max()
        plt.imshow(Z)
        plt.show()


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
        dataset = data.TensorDataset(from_numpy(np.arange(len(self.states))))
        dataloader = data.DataLoader(dataset, batch_size=128, shuffle=True)
        values = np.load("/home/jcoreyes/abstract/rlkit/examples/abstractmdp/values.npy")
        import pdb;
        pdb.set_trace()
        values = from_numpy(np.abs(values))


        for epoch in range(1, max_epochs + 1):
            stats = self.train_epoch(dataloader, epoch, values)

            print(stats)

    def kl(self, dist1, dist2):
        return (dist1 * (torch.log(dist1 + 1e-8) - torch.log(dist2 + 1e-8))).sum(1)

    def entropy(self, dist):
        return -(dist * torch.log(dist + 1e-8)).sum(1)

    def train_epoch(self, dataloader, epoch, values):
        stats = dict([('Loss', 0)])
        states = from_numpy(self.states_np)
        for batch_idx, s1 in enumerate(dataloader):
            s1 = s1[0].long()
            bs = s1.shape[0]
            self.optimizer.zero_grad()

            s2 = from_numpy(np.random.randint(0, len(self.states), bs)).long()
            # Sample s2 where s1 is in same abstract state as s1
            y1 = self.encoder(states[s1])
            y2 = self.encoder(states[s2])

            p1, a1 = y1.max(-1)
            p2, a2 = y2.max(-1)
            match = (a1 == a2) & (s1 != s2)
            distances = values[s2, s1]

            #reward = (distances < 7).float() * 1.0
            reward = -distances * 5e-5
            #import pdb; pdb.set_trace()
            surr_loss = -torch.log(y1[torch.arange(bs), a1] + 1e-8) * reward
            loss = (surr_loss  - 1.0 * self.entropy(y1)) * match.float()

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
    set_gpu_mode(True)
    env = FourRoomsModEnv(gridsize=12)
    a = AbstractMDPVI(env)
    #a.train_vi()
    #a.test_vi()
    a.train(max_epochs=1000)
    a.gen_plot()


