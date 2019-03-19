from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.state_based_goal_experiments import her_dqn_experiment_mincraft
from torch.nn import functional as F
import gym_minecraft
import numpy as np

class BridgeGen:
    def __init__(self):
        self.x = 10
        self.y = 4
        self.z = 10

        self.bank1_lim = (2, 5)
        self.bank2_lim = (6, 9)

    def generate_env(self):
        # air is 0
        # grass is 1
        env = np.zeros((self.y, self.x, self.z)) + 1
        bank1_z = np.random.randint(*self.bank1_lim)
        bank2_z = np.random.randint(*self.bank2_lim)
        bridge_x_pos = np.random.randint(0, self.x)

        env[:, :, bank1_z:bank2_z] = 0
        env[self.y -1, bridge_x_pos, bank1_z:bank2_z] = 2

        return env

if __name__ == "__main__":
    # noinspection PyTypeChecker
    gen = BridgeGen()
    env = gen.generate_env()
    print(env)