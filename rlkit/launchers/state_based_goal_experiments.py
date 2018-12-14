import gym
# Trigger environment registrations
# noinspection PyUnresolvedReferences
import multiworld.envs.mujoco
# noinspection PyUnresolvedReferences
import multiworld.envs.pygame
import rlkit.samplers.rollout_functions as rf
import rlkit.torch.pytorch_util as ptu
from rlkit.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
from rlkit.exploration_strategies.gaussian_strategy import GaussianStrategy
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.launchers.rig_experiments import get_video_save_func
from rlkit.torch.her.her import HerTd3, HerDQN
from rlkit.torch.networks import FlattenMlp, TanhMlpPolicy, MlpPolicy
from rlkit.data_management.obs_dict_replay_buffer import (
    ObsDictRelabelingBuffer
)
import malmoenv
from pathlib import Path
from torch import nn as nn

def her_dqn_experiment_mincraft(variant):
    # if 'env_id' in variant:
    #     env = gym.make(variant['env_id'])
    # else:
    #     env = variant['env_class'](**variant['env_kwargs'])
    env = malmoenv.make()
    xml = Path(variant['mission']).read_text()
    env.init(xml, variant['port'], server='127.0.0.1',
             resync=0, role=0)
    #env.reset()
    observation_key = variant['observation_key']
    desired_goal_key = variant['desired_goal_key']
    variant['algo_kwargs']['her_kwargs']['observation_key'] = observation_key
    variant['algo_kwargs']['her_kwargs']['desired_goal_key'] = desired_goal_key
    if variant.get('normalize', False):
        raise NotImplementedError()

    replay_buffer = ObsDictRelabelingBuffer(
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        **variant['replay_buffer_kwargs']
    )
    obs_dim = env.observation_space.spaces['observation'].low.size
    action_dim = env.action_space.n
    goal_dim = env.observation_space.spaces['desired_goal'].low.size
    exploration_type = variant['exploration_type']
    if exploration_type == 'ou':
        es = OUStrategy(
            action_space=env.action_space,
            **variant['es_kwargs']
        )
    elif exploration_type == 'gaussian':
        es = GaussianStrategy(
            action_space=env.action_space,
            **variant['es_kwargs'],
        )
    elif exploration_type == 'epsilon':
        es = EpsilonGreedy(
            action_space=env.action_space,
            **variant['es_kwargs'],
        )
    else:
        raise Exception("Invalid type: " + exploration_type)
    qf1 = FlattenMlp(
        input_size=obs_dim + goal_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    # qf2 = FlattenMlp(
    #     input_size=obs_dim + action_dim + goal_dim,
    #     output_size=1,
    #     **variant['qf_kwargs']
    # )
    # policy = MlpPolicy(
    #     input_size=obs_dim + goal_dim,
    #     output_size=action_dim,
    #     **variant['policy_kwargs']
    # )
    # exploration_policy = PolicyWrappedWithExplorationStrategy(
    #     exploration_strategy=es,
    #     policy=policy,
    # )
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
    if variant.get("save_video", False):
        rollout_function = rf.create_rollout_function(
            rf.multitask_rollout,
            max_path_length=algorithm.max_path_length,
            observation_key=algorithm.observation_key,
            desired_goal_key=algorithm.desired_goal_key,
        )
        video_func = get_video_save_func(
            rollout_function,
            env,
            #policy,
            variant,
        )
        algorithm.post_epoch_funcs.append(video_func)
    algorithm.to(ptu.device)
    algorithm.train()


def her_td3_experiment(variant):
    if 'env_id' in variant:
        env = gym.make(variant['env_id'])
    else:
        env = variant['env_class'](**variant['env_kwargs'])

    observation_key = variant['observation_key']
    desired_goal_key = variant['desired_goal_key']
    variant['algo_kwargs']['her_kwargs']['observation_key'] = observation_key
    variant['algo_kwargs']['her_kwargs']['desired_goal_key'] = desired_goal_key
    if variant.get('normalize', False):
        raise NotImplementedError()

    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    replay_buffer = ObsDictRelabelingBuffer(
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )
    obs_dim = env.observation_space.spaces['observation'].low.size
    action_dim = env.action_space.low.size
    goal_dim = env.observation_space.spaces['desired_goal'].low.size
    exploration_type = variant['exploration_type']
    if exploration_type == 'ou':
        es = OUStrategy(
            action_space=env.action_space,
            **variant['es_kwargs']
        )
    elif exploration_type == 'gaussian':
        es = GaussianStrategy(
            action_space=env.action_space,
            **variant['es_kwargs'],
        )
    elif exploration_type == 'epsilon':
        es = EpsilonGreedy(
            action_space=env.action_space,
            **variant['es_kwargs'],
        )
    else:
        raise Exception("Invalid type: " + exploration_type)
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim + goal_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim + goal_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim + goal_dim,
        output_size=action_dim,
        **variant['policy_kwargs']
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    algorithm = HerTd3(
        env,
        qf1=qf1,
        qf2=qf2,
        policy=policy,
        exploration_policy=exploration_policy,
        replay_buffer=replay_buffer,
        **variant['algo_kwargs']
    )
    if variant.get("save_video", False):
        rollout_function = rf.create_rollout_function(
            rf.multitask_rollout,
            max_path_length=algorithm.max_path_length,
            observation_key=algorithm.observation_key,
            desired_goal_key=algorithm.desired_goal_key,
        )
        video_func = get_video_save_func(
            rollout_function,
            env,
            policy,
            variant,
        )
        algorithm.post_epoch_funcs.append(video_func)
    algorithm.to(ptu.device)
    algorithm.train()
