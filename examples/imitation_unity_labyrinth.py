import pathlib
from tabnanny import verbose
# import pickle
import tempfile
# import os

import stable_baselines3 as sb3
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.common.utils import obs_as_tensor

from imitation.algorithms import adversarial, bc
from imitation.rewards import reward_nets
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util import logger
from imitation.algorithms.adversarial.airl import AIRL
from imitation.util.networks import RunningNorm

import sys, os
sys.path.append('..')
from utils.demo_loader import load_and_parse_unity_demo
from Environments.unity_labyrinth import build_unity_labyrinth_env


tempdir = tempfile.TemporaryDirectory(prefix="quickstart")
tempdir_path = pathlib.Path(tempdir.name)
print(f"All Tensorboards and logging are being written inside {tempdir_path}/.")

# Load the unity demo file
file_path = os.path.abspath('../demos/labyrinth/IRLnonlava.demo')
sequence_length = 3382 #IRLnonlava.demo
sequence_length = 2513 #IRLlava.demo
# sequence_length = 1797 #wholelabyrinthhu_0.demo
sequence_length = 1435 #c1.demo
# sequence_length = 1217 # room4hugleft.demo

transitions = load_and_parse_unity_demo(demo_file=file_path, 
                                        sequence_length=sequence_length)

# Load the unity labyrinth environment.
env_settings = {
    'time_scale' : 99.0,
}

env, side_channels = build_unity_labyrinth_env()
side_channels['engine_config_channel'].set_configuration_parameters(
                                        time_scale=env_settings['time_scale'])

# #################### AIRL
# Wrap the environemnt in a dummy vectorized environment.
venv = DummyVecEnv([lambda: env])

learner = sb3.PPO(
    env=env,
    policy="MlpPolicy",
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0003,
    n_epochs=10,
)
reward_net = BasicRewardNet(
    venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm
)
airl_trainer = AIRL(
    demonstrations=transitions,
    demo_batch_size=1024,
    gen_replay_buffer_capacity=2048,
    n_disc_updates_per_round=4,
    venv=venv,
    gen_algo=learner,
    reward_net=reward_net,
    allow_variable_horizon=True,
)

# Set the environment task to be the overall composite task
side_channels['custom_side_channel'].send_string('1,1')

# learner_rewards_before_training, _ = evaluate_policy(
#     learner, venv, 100, return_episode_rewards=True
# )
airl_trainer.train(300000)  # Note: set to 300000 for better results
# learner_rewards_after_training, _ = evaluate_policy(
#     learner, venv, 100, return_episode_rewards=True
# )
predict = airl_trainer.gen_algo.policy

#######################################################################
# Train BC on expert data.
# BC also accepts as `demonstrations` any PyTorch-style DataLoader that iterates over
# dictionaries containing observations and actions.

# ppo_model = sb3.PPO("MlpPolicy", 
#                             env, 
#                             verbose=1,
#                             n_steps=512,
#                             batch_size=64,
#                             gae_lambda=0.95,
#                             gamma=0.99,
#                             n_epochs=10,
#                             ent_coef=0.0,
#                             learning_rate=2.5e-4,
#                             clip_range=0.2)

# bc_logger = logger.configure(tempdir_path / "BC/")
# bc_trainer = bc.BC(
#     observation_space=env.observation_space,
#     action_space=env.action_space,
#     # policy=ppo_model.policy,
#     #expert_data=transitions,
#     demonstrations=transitions,
#     # custom_logger=bc_logger,
# )
# # Set the environment task to be the overall composite task
# side_channels['custom_side_channel'].send_string('1,1')

# # Train to match expert demonstrations
# bc_trainer.train(n_epochs=15)

# # # ppo_model.learn(1000)

# predict = bc_trainer.policy.predict

######################################################################
# ppo_policy = sb3.PPO("MlpPolicy", venv, verbose=1, n_steps=500)

# # Train GAIL on expert data.
# # GAIL, and AIRL also accept as `expert_data` any Pytorch-style DataLoader that
# # iterates over dictionaries containing observations, actions, and next_observations.
# logger.configure(tempdir_path / "GAIL/")
# # gail_reward_net = reward_nets.BasicRewardNet(
# #     observation_space=venv.observation_space,
# #     action_space=venv.action_space,
# # )
# # discrim_kwargs = {'discrim_net' : gail_reward_net}
# gail_trainer = adversarial.GAIL(
#     venv,
#     expert_data=transitions,
#     expert_batch_size=32,
#     gen_algo=ppo_policy,
#     # discrim_kwargs=discrim_kwargs
# )

# # Set the environment task to be the overall composite task
# side_channels['custom_side_channel'].send_string('1,1')

# gail_trainer.train(total_timesteps=300000)
# predict = gail_trainer.gen_algo.policy.predict

# Test the results.
n_episodes = 10
n_steps = 500

side_channels['engine_config_channel']\
            .set_configuration_parameters(time_scale=1.0)

# Set the environment task to be the overall composite task
side_channels['custom_side_channel'].send_string('1,1')

print('demonstrating behavior.')
for episode_ind in range(n_episodes):
    obs = env.reset()
    # self.reset(side_channels)
    for step in range(n_steps):
        action, _states = predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if done:
            break