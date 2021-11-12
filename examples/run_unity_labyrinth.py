import os, sys
sys.path.append('..')

from Environments.unity_labyrinth import build_unity_labyrinth_env
import numpy as np
from Controllers.unity_labyrinth_controller import UnityLabyrinthController
from Controllers.unity_meta_controller import MetaController
import pickle
from datetime import datetime
import torch
import random
from MDP.general_high_level_mdp import HLMDP
from utils.results_saver import Results

# Setup and create the environment

env_settings = {
    'time_scale' : 99.0,
}

env, side_channels = build_unity_labyrinth_env()
side_channels['engine_config_channel'].set_configuration_parameters(
                                        time_scale=env_settings['time_scale'])

prob_threshold = 0.95 # Desired probability of reaching the final goal
training_iters = 5e4 # 5e4
num_rollouts = 100 # 100
n_steps_per_rollout = 500
max_timesteps_per_component = 2e5

# Set the load directory (if loading pre-trained sub-systems) 
# or create a new directory in which to save results

load_folder_name = ''
save_learned_controllers = True

experiment_name = 'unity_labyrinth'

base_path = os.path.abspath(os.path.curdir)
string_ind = base_path.find('examples')
assert(string_ind >= 0)
base_path = base_path[0:string_ind]
base_path = os.path.join(base_path, 'data', 'saved_controllers')

load_dir = os.path.join(base_path, load_folder_name)

if load_folder_name == '':
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    rseed = int(now.time().strftime('%H%M%S'))
    save_path = os.path.join(base_path, dt_string + '_' + experiment_name)
else:
    save_path = os.path.join(base_path, load_folder_name)

if save_learned_controllers and not os.path.isdir(save_path):
    os.mkdir(save_path)

# Create the list of partially instantiated sub-systems
controller_list = []

if load_folder_name == '':
    for i in range(12):
        controller_list.append(UnityLabyrinthController(i, env, env_settings=env_settings))
else:
    for controller_dir in os.listdir(load_dir):
        controller_load_path = os.path.join(load_dir, controller_dir)
        if os.path.isdir(controller_load_path):
            controller = UnityLabyrinthController(0, env, load_dir=controller_load_path)
            controller_list.append(controller)

    # re-order the controllers by index
    reordered_list = []
    for i in range(len(controller_list)):
        for controller in controller_list:
            if controller.controller_ind == i:
                reordered_list.append(controller)
    controller_list = reordered_list

# Create or load object to store the results
if load_folder_name == '':
    results = Results(controller_list, 
                        env_settings, 
                        prob_threshold, 
                        training_iters, 
                        num_rollouts, 
                        random_seed=rseed)
else:
    results = Results(load_dir=load_dir)
    rseed = results.data['random_seed']

torch.manual_seed(rseed)
random.seed(rseed)
np.random.seed(rseed)

print('Random seed: {}'.format(results.data['random_seed']))

for controller_ind in range(len(controller_list)):
    controller = controller_list[controller_ind]
    # Evaluate initial performance of controllers (they haven't learned 
    # anything yet so they will likely have no chance of success.)
    controller.eval_performance(env, 
                                side_channels['custom_side_channel'], 
                                n_episodes=10,
                                n_steps=n_steps_per_rollout)
    print('Controller {} achieved prob succes: {}'.format(controller_ind, 
                                                controller.get_success_prob()))

    # Save learned controller
    if save_learned_controllers:
        controller_save_path = \
            os.path.join(save_path, 'controller_{}'.format(controller_ind))
        controller.save(controller_save_path)

results.update_training_steps(0)
results.update_controllers(controller_list)
results.save(save_path)

# Construct high-level MDP and solve for the max reach probability
S = np.arange(-1, 11)
A = np.arange(len(controller_list))
s_i = 0
s_goal = 8
s_fail = -1

successor_map = {
    (0,0) : 2,
    (0,1) : 1,
    (1,2) : 3,
    (1,3) : 5,
    (2,4) : 9,
    (9,5) : 10,
    (3,6) : 4,
    (4,7) : 3,
    (5,8) : 6,
    (10,9): 8,
    (6,10): 7,
    (7,11): 8,
}

hlmdp = HLMDP(S, A, s_i, s_goal, s_fail, controller_list, successor_map)
policy, reach_prob, feasible_flag = hlmdp.solve_max_reach_prob_policy()

# Construct a meta-controller and emprirically evaluate it.
meta_controller = MetaController(policy, hlmdp, side_channels)
meta_success_rate = meta_controller.eval_performance(env,
                                                    side_channels,
                                                    n_episodes=num_rollouts, 
                                                    n_steps=n_steps_per_rollout)
meta_controller.unsubscribe_meta_controller(side_channels)

# Save the results
results.update_composition_data(meta_success_rate, num_rollouts, policy, reach_prob)
results.save(save_path)

# Main loop of iterative compositional reinforcement learning

total_timesteps = training_iters

while reach_prob < prob_threshold:

    # Solve the HLM biliniear program to obtain sub-task specifications.
    optimistic_policy, \
        required_reach_probs, \
            optimistic_reach_prob, \
                feasible_flag = \
                    hlmdp.solve_low_level_requirements_action(prob_threshold, 
                    max_timesteps_per_component=max_timesteps_per_component)

    if not feasible_flag:
        print(required_reach_probs)

    # Print the empirical sub-system estimates and the sub-system 
    # specifications to terminal
    for controller_ind in range(len(hlmdp.controller_list)):
        controller = hlmdp.controller_list[controller_ind]
        print('Sub-task: {}, \
                Achieved success prob: {}, Required success prob: {}'\
                    .format(controller_ind, 
                            controller.get_success_prob(), 
                            controller.data['required_success_prob']))

    # Decide which sub-system to train next.
    performance_gaps = []
    for controller_ind in range(len(hlmdp.controller_list)):
        controller = hlmdp.controller_list[controller_ind]
        performance_gaps.append(controller.data['required_success_prob'] - \
                                controller.get_success_prob())

    largest_gap_ind = np.argmax(performance_gaps)
    controller_to_train = hlmdp.controller_list[largest_gap_ind]

    # Train the sub-system and empirically evaluate its performance
    print('Training controller {}'.format(largest_gap_ind))
    controller_to_train.learn(side_channels['custom_side_channel'], 
                                total_timesteps=total_timesteps)
    print('Completed training controller {}'.format(largest_gap_ind))
    controller_to_train.eval_performance(env, 
                                        side_channels['custom_side_channel'], 
                                        n_episodes=num_rollouts,
                                        n_steps=n_steps_per_rollout)

    # Save learned controller
    if save_learned_controllers:
        controller_save_path = os.path.join(save_path, 
                                    'controller_{}'.format(largest_gap_ind))
        if not os.path.isdir(controller_save_path):
            os.mkdir(controller_save_path)
        controller_to_train.save(controller_save_path)

    # Solve the HLM for the meta-policy maximizing reach probability
    policy, reach_prob, feasible_flag = hlmdp.solve_max_reach_prob_policy()

    # Construct a meta-controller with this policy and empirically evaluate its performance
    meta_controller = MetaController(policy, hlmdp, side_channels)
    meta_success_rate = meta_controller.eval_performance(env,
                                                        side_channels, 
                                                        n_episodes=num_rollouts, 
                                                        n_steps=n_steps_per_rollout)
    meta_controller.unsubscribe_meta_controller(side_channels)

    # Save results
    results.update_training_steps(total_timesteps)
    results.update_controllers(hlmdp.controller_list)
    results.update_composition_data(meta_success_rate, num_rollouts, policy, reach_prob)
    results.save(save_path)

# Once the loop has been completed, construct a meta-controller and visualize its performance

meta_controller = MetaController(policy, hlmdp, side_channels)
print('evaluating performance of meta controller')
meta_success_rate = meta_controller.eval_performance(env,
                                                    side_channels,
                                                    n_episodes=num_rollouts, 
                                                    n_steps=n_steps_per_rollout)
meta_controller.unsubscribe_meta_controller(side_channels)
print('Predicted success prob: {}, \
    empirically measured success prob: {}'.format(reach_prob, meta_success_rate))

# 
n_episodes = 5
n_steps_per_rollout = 1000
render = True
meta_controller = MetaController(policy, hlmdp, side_channels)
meta_controller.demonstrate_capabilities(env, 
                                        side_channels,
                                        n_episodes=n_episodes, 
                                        n_steps=n_steps_per_rollout, 
                                        render=render)
meta_controller.unsubscribe_meta_controller(side_channels)