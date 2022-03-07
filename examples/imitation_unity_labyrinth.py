import pathlib
# import pickle
import tempfile
# import os

import stable_baselines3 as sb3
# from stable_baselines3.common.utils import obs_as_tensor

from imitation.algorithms import adversarial, bc
from imitation.util import logger

import sys, os
sys.path.append('..')
from utils.demo_loader import load_and_parse_unity_demo
from Environments.unity_labyrinth import build_unity_labyrinth_env


tempdir = tempfile.TemporaryDirectory(prefix="quickstart")
tempdir_path = pathlib.Path(tempdir.name)
print(f"All Tensorboards and logging are being written inside {tempdir_path}/.")

# Load the unity demo file
file_path = os.path.abspath('../demos/labyrinth/IRLnonlava.demo')
transitions = load_and_parse_unity_demo(demo_file=file_path, sequence_length=2513)

# Load the unity labyrinth environment.
env_settings = {
    'time_scale' : 99.0,
}

env, side_channels = build_unity_labyrinth_env()
side_channels['engine_config_channel'].set_configuration_parameters(
                                        time_scale=env_settings['time_scale'])

# Train BC on expert data.
# BC also accepts as `demonstrations` any PyTorch-style DataLoader that iterates over
# dictionaries containing observations and actions.
bc_logger = logger.configure(tempdir_path / "BC/")
bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    expert_data=transitions,
    # custom_logger=bc_logger,
)
bc_trainer.train(n_epochs=1)

print(bc_trainer)

# Test the results.
n_episodes = 10
n_steps = 3500

side_channels['engine_config_channel']\
            .set_configuration_parameters(time_scale=1.0)

# Set the environment task to be the overall composite task
side_channels['custom_side_channel'].send_string('-1,-1')

for episode_ind in range(n_episodes):
    obs = env.reset()
    # self.reset(side_channels)
    for step in range(n_steps):
        action, _states = bc_trainer.policy.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if done:
            break