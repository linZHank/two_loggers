#! /usr/bin/env python
"""
Training two logger robots escaping a cell with Deep Q-network (DQN)
DQN is a model free, off policy, reinforcement learning algorithm (https://deepmind.com/research/dqn/)
Author: LinZHanK (linzhank@gmail.com)

Train new models example:
    python double_escape_dqn_train.py --num_episodes 20000 --num_steps 400 --normalize --update_step 10000 --time_bonus -0.0025 --wall_bonus -0.025 --door_bonus 0 --success_bonus 1
Continue training models example:
    python double_escape_dqn_train.py --source '2019-07-17-17-57' --num_episodes 100
"""
from __future__ import absolute_import, division, print_function

import sys
import os
import time
from datetime import datetime
import numpy as np
import random
import math
import tensorflow as tf
import rospy
import pickle

from envs.double_escape_task_env import DoubleEscapeEnv
from utils import data_utils, double_utils, tf_utils
from utils.data_utils import bcolors
from agents.dqn import DQNAgent

import pdb


if __name__ == "__main__":
    # create argument parser
    args = double_utils.get_args()
    # make an instance from env class
    env = DoubleEscapeEnv()
    env.reset()
    # create training parameters
    date_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    if not args.source: # source is empty, create new params
        complete_episodes = 0
        train_params = double_utils.create_train_params(date_time, complete_episodes, args.source, args.normalize, args.num_episodes, args.num_steps, args.time_bonus, args.wall_bonus, args.door_bonus, args.success_bonus)
        # init agent parameters
        dim_state = len(double_utils.obs_to_state(env.observation, "all"))
        actions = np.array([np.array([1, -1]), np.array([1, 1])])
        layer_sizes = [256, 256]
        gamma = 0.99
        learning_rate = 1e-3
        batch_size = 2048
        memory_cap = 1000000
        update_step = 8000
        decay_period = args.num_episodes/4
        final_eps = 5e-3
        agent_params_0 = double_utils.create_agent_params(dim_state, actions, layer_sizes, gamma, learning_rate, batch_size, memory_cap, update_step, decay_period, final_eps)
        agent_params_0['update_counter'] = 0
        agent_params_1 = agent_params_0
        # instantiate new agents
        agent_0 = DQNAgent(agent_params_0)
        model_path_0 = os.path.dirname(sys.path[0])+"/saved_models/double_escape/dqn/"+date_time+"/agent_0/model.h5"
        agent_1 = DQNAgent(agent_params_1)
        model_path_1 = os.path.dirname(sys.path[0])+"/saved_models/double_escape/dqn/"+date_time+"/agent_1/model.h5"
        assert os.path.dirname(os.path.dirname(model_path_0)) == os.path.dirname(os.path.dirname(model_path_1))
        # init returns and losses storage
        train_params['ep_returns'] = []
        agent_params_0['ep_losses'] = []
        agent_params_0['ep_losses'] = []
        # init random starting poses
        train_params['pose_buffer'] = double_utils.create_pose_buffer(train_params['num_episodes']+1)
        # init success count
        train_params['success_count'] = 0
    else: # source is not empty, load params
        model_load_dir = os.path.dirname(sys.path[0])+"/saved_models/double_escape/dqn/"+args.source
        # load train parameters
        train_params_path = os.path.join(model_load_dir, "train_params.pkl")
        with open(train_params_path, 'rb') as f:
            train_params = pickle.load(f)
        train_params['source'] = args.source
        train_params["date_time"] = date_time
        ep_returns = train_params['ep_returns']
        # load agents parameters
        agent_params_path_0 = os.path.join(model_load_dir,"agent_0/agent0_parameters.pkl")
        with open(agent_params_path_0, 'rb') as f:
            agent_params_0 = pickle.load(f) # load agent_0 model
        agent_params_path_1 = os.path.join(model_load_dir,"agent_1/agent1_parameters.pkl")
        with open(agent_params_path_1, 'rb') as f:
            agent_params_1 = pickle.load(f) # load agent_1 model
        # load dqn models & memory buffers
        agent_0 = DQNAgent(agent_params_0)
        model_path_0 = os.path.dirname(sys.path[0])+"/saved_models/double_escape/dqn/"+date_time+"/agent_0/model.h5"
        agent_0.load_model(os.path.join(model_load_dir, "agent_0/model.h5"))
        agent_1 = DQNAgent(agent_params_1)
        model_path_1 = os.path.dirname(sys.path[0])+"/saved_models/double_escape/dqn/"+date_time+"/agent_1/model.h5"
        agent_1.load_model(os.path.join(model_load_dir, "agent_1/model.h5"))
        # load means and stds

    # init very first episode and step
    obs, _ = env.reset(train_params['pose_buffer'][0])
    state_0 = double_utils.obs_to_state(obs, "all")
    state_1 = double_utils.obs_to_state(obs, "all")
    if not args.source:
        # new means and stds
        mean_0 = state_0 # states average
        std_0 = np.zeros(agent_params_0["dim_state"])+1e-8 # n*Var
        mean_1 = state_1 # states average
        std_1 = np.zeros(agent_params_1["dim_state"])+1e-8 # n*Var
    else:
        mean_0 = agent_params_0['mean']
        std_0 = agent_params_0['std']
        mean_1 = agent_params_1['mean']
        std_1 = agent_params_1['std']
    # timing
    start_time = time.time()
    for ep in range(train_params['complete_episodes'], train_params['num_episodes']):
        # check simulation crash
        if sum(np.isnan(state_0)) >= 1 or sum(np.isnan(state_1)) >= 1:
            print(bcolors.FAIL, "Simulation Crashed", bcolors.ENDC)
            train_params['complete_episodes'] = ep
            break # terminate main loop if simulation crashed
        epsilon_0 = agent_0.linearly_decaying_epsilon(decay_period=agent_params_0['decay_period'], episode=ep)
        epsilon_1 = agent_1.linearly_decaying_epsilon(decay_period=agent_params_1['decay_period'], episode=ep)
        print("epsilon_0: {}, epsilon_1: {}".format(epsilon_0, epsilon_1))
        # theta_0, theta_1 = random.uniform(-math.pi, math.pi), random.uniform(-math.pi, math.pi)
        done, ep_rewards, loss_vals_0, loss_vals_1 = False, [], [], []
        for st in range(train_params["num_steps"]):
            # check simulation crash
            if sum(np.isnan(state_0)) >= 1 or sum(np.isnan(state_1)) >= 1:
                print(bcolors.FAIL, "Simulation Crashed", bcolors.ENDC)
                break # tbreakout loop if gazebo crashed
            # normalize states
            if train_params['normalize']:
                norm_state_0 = tf_utils.normalize(state_0, mean_0, std_0)
                norm_state_1 = tf_utils.normalize(state_1, mean_1, std_1)
                print(bcolors.WARNING, "States normalize: {}".format((norm_state_0, norm_state_1)), bcolors.ENDC)
            else:
                norm_state_0 = state_0
                norm_state_1 = state_1
            agent0_acti = agent_0.epsilon_greedy(norm_state_0)
            agent0_action = agent_0.actions[agent0_acti]
            agent1_acti = agent_1.epsilon_greedy(norm_state_1)
            agent1_action = agent_1.actions[agent1_acti]
            obs, rew, done, info = env.step(agent0_action, agent1_action)
            next_state_0 = double_utils.obs_to_state(obs, "all")
            next_state_1 = double_utils.obs_to_state(obs, "all")
            # compute incremental mean and std
            inc_mean_0 = tf_utils.increment_mean(mean_0, next_state_0, (ep+1)*(st+1)+1)
            inc_std_0 = tf_utils.increment_std(std_0, mean_0, inc_mean_0, next_state_0, (ep+1)*(st+1)+1)
            inc_mean_1 = tf_utils.increment_mean(mean_1, next_state_1, (ep+1)*(st+1)+1)
            inc_std_1 = tf_utils.increment_std(std_1, mean_1, inc_mean_1, next_state_1, (ep+1)*(st+1)+1)
            # update mean and std
            mean_0, std_0, mean_1, std_1 = inc_mean_0, inc_std_0, inc_mean_1, inc_std_1
            agent_params_0['mean'] = mean_0
            agent_params_0['std'] = std_0
            agent_params_0['mean'] = mean_1
            agent_params_1['std'] = std_1
            # normalize next states
            if train_params['normalize']:
                norm_next_state_0 = tf_utils.normalize(next_state_0, mean_0, std_0)
                norm_next_state_1 = tf_utils.normalize(next_state_1, mean_1, std_1)
                print(bcolors.WARNING, "Next states normalized: {}".format((norm_next_state_0, norm_next_state_1)), bcolors.ENDC)
            else:
                norm_next_state_0 = next_state_0
                norm_next_state_1 = next_state_1
            # adjust reward based on bonus options
            rew, done = double_utils.adjust_reward(train_params, env)
            ep_rewards.append(rew)
            train_params['success_count'] += env.success_count
            # rew, done = double_utils.adjust_reward(hyp_params, env, agent)
            print(
                bcolors.OKGREEN,
                "Episode: {}, Step: {}: \nstate_0: {}, action0: {}->{}, next state_0 : {} \nstate_1: {}, action1: {}->{}, next state_1: {}, reward/episodic_return: {}/{}, status: {}, succeeded: {}".format(
                    ep+1,
                    st+1,
                    state_0,
                    agent0_acti,
                    agent0_action,
                    next_state_0,
                    state_1,
                    agent1_acti,
                    agent1_action,
                    next_state_1,
                    rew,
                    sum(ep_rewards),
                    info["status"],
                    train_params['success_count']
                ),
                bcolors.ENDC
            )
            # store transitions
            if not info["status"] == "blew":
                agent_0.replay_memory.store((norm_state_0, agent0_acti, rew, done, norm_next_state_0))
                agent_1.replay_memory.store((norm_state_1, agent1_acti, rew, done, norm_next_state_1))
                print(bcolors.OKBLUE, "transition saved to memory", bcolors.ENDC)
            else:
                print(bcolors.FAIL, "model blew up, transition not saved", bcolors.ENDC)
            agent_0.train()
            loss_vals_0.append(agent_0.loss_value)
            agent_1.train()
            loss_vals_1.append(agent_1.loss_value)
            state_0 = next_state_0
            state_1 = next_state_1
            agent_params_0['update_counter'] += 1
            agent_params_1['update_counter'] += 1
            if not agent_params_0['update_counter'] % agent_params_0['update_step']:
                agent_0.qnet_stable.set_weights(agent_0.qnet_active.get_weights())
                print(bcolors.BOLD, "agent_0 Q-net weights updated!", bcolors.ENDC)
            if not agent_params_1['update_counter'] % agent_params_1['update_step']:
                agent_1.qnet_stable.set_weights(agent_1.qnet_active.get_weights())
                print(bcolors.BOLD, "agent_1 Q-net weights updated!", bcolors.ENDC)
            if done:
                train_params['complete_episodes'] += 1
                break
        train_params['ep_returns'].append(sum(ep_rewards))
        agent_params_0['ep_losses'].append(sum(loss_vals_0)/len(loss_vals_0))
        agent_params_1['ep_losses'].append(sum(loss_vals_1)/len(loss_vals_1))
        agent_0.save_model(model_path_0)
        agent_1.save_model(model_path_1)
        obs, _ = env.reset(train_params['pose_buffer'][ep+1])
        state_0 = double_utils.obs_to_state(obs, "all")
        state_1 = double_utils.obs_to_state(obs, "all")
    # time training
    end_time = time.time()
    train_dur = end_time - start_time
    env.reset()

    # save replay buffer memories
    agent_0.save_memory(model_path_0)
    agent_1.save_memory(model_path_1)
    # save agent parameters
    data_utils.save_pkl(content=agent_params_0, fdir=os.path.dirname(model_path_0), fname="agent0_parameters.pkl")
    data_utils.save_pkl(content=agent_params_1, fdir=os.path.dirname(model_path_1), fname="agent1_parameters.pkl")
    # save returns and losses
    # data_utils.save_pkl(content=ep_returns, fdir=os.path.dirname(os.path.dirname(model_path_0)), fname="episodic_returns.pkl")
    # data_utils.save_pkl(content=ep_losses_0, fdir=os.path.dirname(model_path_0), fname="episodic_average_losses.pkl")
    # data_utils.save_pkl(content=ep_losses_1, fdir=os.path.dirname(model_path_1), fname="episodic_average_losses.pkl")

    # save results
    train_info = train_params
    train_info["train_dur"] = train_dur
    train_info["agent0_learning_rate"] = agent_params_0["learning_rate"]
    train_info["agent0_state_dimension"] = agent_params_0["dim_state"]
    train_info["agent0_action_options"] = agent_params_0["actions"]
    train_info["agent0_layer_sizes"] = agent_params_0["layer_sizes"]
    train_info["agent1_learning_rate"] = agent_params_1["learning_rate"]
    train_info["agent1_state_dimension"] = agent_params_1["dim_state"]
    train_info["agent1_action_options"] = agent_params_1["actions"]
    train_info["agent1_layer_sizes"] = agent_params_1["layer_sizes"]
    data_utils.save_pkl(content=train_params, fdir=os.path.dirname(os.path.dirname(model_path_0)), fname="train_params.pkl")
    data_utils.save_csv(content=train_info, fdir=os.path.dirname(os.path.dirname(model_path_0)), fname="train_information.csv")
    data_utils.save_pkl(content=train_info, fdir=os.path.dirname(os.path.dirname(model_path_0)), fname="train_info.pkl")

    # plot episodic returns
    data_utils.plot_returns(returns=train_params['ep_returns'], mode=0, save_flag=True, fdir=os.path.dirname(os.path.dirname(model_path_0)))
    # plot accumulated returns
    data_utils.plot_returns(returns=train_params['ep_returns'], mode=1, save_flag=True, fdir=os.path.dirname(os.path.dirname(model_path_0)))
    # plot averaged return
    data_utils.plot_returns(returns=train_params['ep_returns'], mode=2, save_flag=True,
    fdir=os.path.dirname(os.path.dirname(model_path_0)))