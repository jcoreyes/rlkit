from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.conv_networks import CNN
from rlkit.torch.her.her import HerDQN
from torch import nn
import rlkit.torch.pytorch_util as ptu
import gym_minigrid
from torch.nn import functional as F
import gym
import numpy as np


def her_dqn_experiment_minigrid(variant):
    env = gym.make(variant['env_id'])

    observation_key = variant['observation_key']
    desired_goal_key = variant['desired_goal_key']
    variant['algo_kwargs']['her_kwargs']['observation_key'] = observation_key
    variant['algo_kwargs']['her_kwargs']['desired_goal_key'] = desired_goal_key
    # if variant.get('normalize', False):
    #     raise NotImplementedError()

    replay_buffer = ObsDictRelabelingBuffer(
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        internal_keys=['agent_pos'],
        **variant['replay_buffer_kwargs']
    )
    obs_shape = env.obs_shape
    action_dim = env.action_space.n
    #goal_shape = env.observation_space.spaces['desired_goal'].shape

    qf1 = CNN(obs_shape[0], obs_shape[1], obs_shape[2],
              output_size=action_dim,
              kernel_sizes=[2, 2],
              n_channels=[16, 32],
              strides=[1, 1],
              paddings=np.zeros(2, dtype=np.int64),
              added_fc_input_size=env.add_input_dim*2,
              hidden_sizes=(128, 128),
              )
    algorithm = HerDQN(
        env,
        training_env=env,
        qf=qf1,
        #qf2=qf2,
        #policy=policy,
        #exploration_policy=exploration_policy,
        replay_buffer=replay_buffer,
        qf_criterion=nn.MSELoss(),
        **variant['algo_kwargs']
    )

    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_kwargs=dict(
            collection_mode='online',
            num_updates_per_epoch=1000,
            dqn_kwargs=dict(
                num_epochs=5000,
                num_steps_per_epoch=200,
                num_steps_per_eval=99,
                batch_size=128,
                max_path_length=1000,
                discount=0.99,
                epsilon=0.05,
                tau=0.002,
                hard_update_period=1000,
                save_environment=False,
            ),
            her_kwargs=dict(
                observation_key='state_observation',
                desired_goal_key='desired_goal',
            ),
        ),
        replay_buffer_kwargs=dict(
            max_size=int(1E6),
            fraction_goals_rollout_goals=0.2,
            fraction_goals_env_goals=0.2,
            #fraction_goals_rollout_goals=0.2, # sample from rollout
            #fraction_goals_env_goals=0.2,# sample from replay buffer
        ),                               # rest of goals are true goal
        qf_kwargs=dict(
            hidden_sizes=[128, 128],
        ),
        version='normal',
        exploration_type='epsilon',
        observation_key='state_observation',
        desired_goal_key='desired_goal',
        init_camera=None,
        do_state_exp=True,

        save_video=False,
        imsize=84,

        snapshot_mode='gap_and_last',
        snapshot_gap=10,

        port=9001,
        env_kwargs=dict(
            render_onscreen=False,
            ball_radius=1,
            images_are_rgb=True,
            show_goal=False,
        ),

        algorithm='HerDQN',
        env_id='MiniGridMod-Bridge-v0'
    )

    n_seeds = 1
    mode = 'here_no_doodad'
    exp_prefix = 'rlkit-minigrid-bridge'

    for _ in range(n_seeds):
        run_experiment(
            her_dqn_experiment_minigrid,
            exp_prefix=exp_prefix,
            mode=mode,
            variant=variant,
            use_gpu=True,  # Turn on if you have a GPU
        )
