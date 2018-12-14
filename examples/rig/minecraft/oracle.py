from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v2
from multiworld.envs.pygame.point2d import Point2DWallEnv
from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.state_based_goal_experiments import her_dqn_experiment_mincraft
from torch.nn import functional as F

if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_kwargs=dict(
            dqn_kwargs=dict(
                num_epochs=500,
                num_steps_per_epoch=100,
                num_steps_per_eval=100,
                batch_size=128,
                max_path_length=200,
                discount=0.99,
                epsilon=0.2,
                tau=0.001,
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
            fraction_goals_rollout_goals=0.1,
            fraction_goals_env_goals=0.5,
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        version='normal',
        es_kwargs=dict(
            #max_sigma=.2,
            prob_random_action=0.05,
        ),
        exploration_type='epsilon',
        observation_key='state_observation',
        desired_goal_key='desired_goal',
        init_camera=sawyer_pusher_camera_upright_v2,
        do_state_exp=True,

        save_video=False,
        imsize=84,

        snapshot_mode='gap_and_last',
        snapshot_gap=10,

        mission='/home/jcoreyes/abstract/malmo/MalmoEnv/missions/wallbuilder.xml',
        port=9000,
        env_kwargs=dict(
            render_onscreen=False,
            ball_radius=1,
            images_are_rgb=True,
            show_goal=False,
        ),

        algorithm='Oracle',
    )

    n_seeds = 1
    mode = 'here_no_doodad'
    exp_prefix = 'rlkit-pointmass-oracle'

    for _ in range(n_seeds):
        run_experiment(
            her_dqn_experiment_mincraft,
            exp_prefix=exp_prefix,
            mode=mode,
            variant=variant,
            # use_gpu=True,  # Turn on if you have a GPU
        )
