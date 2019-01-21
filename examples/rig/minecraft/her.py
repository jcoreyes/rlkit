from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.state_based_goal_experiments import her_dqn_experiment_mincraft
from torch.nn import functional as F
import gym_minecraft

if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_kwargs=dict(
            collection_mode='online',
            num_updates_per_epoch=1000,
            dqn_kwargs=dict(
                num_epochs=5000,
                num_steps_per_epoch=400,
                num_steps_per_eval=199,
                batch_size=128,
                max_path_length=1000,
                discount=0.99,
                epsilon=0.2,
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
            fraction_goals_rollout_goals=0.0,
            fraction_goals_env_goals=1.0,
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
        env_id='MinecraftWallBuilder-v0'
    )

    n_seeds = 1
    mode = 'here_no_doodad'
    exp_prefix = 'rlkit-wallbuilder-oracle'

    for _ in range(n_seeds):
        run_experiment(
            her_dqn_experiment_mincraft,
            exp_prefix=exp_prefix,
            mode=mode,
            variant=variant,
            use_gpu=True,  # Turn on if you have a GPU
        )
