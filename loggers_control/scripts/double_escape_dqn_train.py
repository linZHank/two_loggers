#! /usr/bin/env python
"""
Training two logger robots escaping a cell with Deep Q-network (DQN)
DQN is a model free, off policy, reinforcement learning algorithm (https://deepmind.com/research/dqn/)
Author: LinZHanK (linzhank@gmail.com)
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

from envs.double_escape_task_env import DoubleEscapeEnv
from utils import data_utils, double_utils, tf_utils
from utils.data_utils import bcolors
from agents.dqn import DQNAgent

import pdb


if __name__ == "__main__":
    # create argument parser
    args = data_utils.get_args()
    # make an instance from env class
    env = DoubleEscapeEnv()
    env.reset()
    agent0_params = {}
    agent1_params = {}
    train_params = {}
    # agent_0 parameters
    agent0_params["dim_state"] = len(double_utils.obs_to_state(env.observation, "all"))
    agent0_params["actions"] = np.array([np.array([1, -1]), np.array([1, 1])])
    agent0_params["layer_sizes"] = args.layer_sizes
    agent0_params["gamma"] = args.gamma
    agent0_params["learning_rate"] = args.learning_rate
    agent0_params["batch_size"] = args.batch_size
    agent0_params["memory_cap"] = args.memory_cap
    agent0_params["update_step"] = args.update_step
    # agent_1 parameters
    agent1_params["dim_state"] = len(double_utils.obs_to_state(env.observation, "all"))
    agent1_params["actions"] = agent0_params["actions"]
    agent1_params["layer_sizes"] = args.layer_sizes
    agent1_params["gamma"] = args.gamma
    agent1_params["learning_rate"] = args.learning_rate
    agent1_params["batch_size"] = args.batch_size
    agent1_params["memory_cap"] = args.memory_cap
    agent1_params["update_step"] = args.update_step
    # training parameters
    if args.datetime:
        train_params["datetime"] = args.datetime
    else:
        train_params["datetime"] = datetime.now().strftime("%Y-%m-%d-%H-%M")
    train_params["num_episodes"] = args.num_episodes
    train_params["num_steps"] = args.num_steps
    train_params["time_bonus"] = -1./train_params["num_steps"]
    train_params["success_bonus"] = 0
    train_params["wall_bonus"] = -10./train_params["num_steps"]
    train_params["door_bonus"] = 0
    # instantiate agents
    agent_0 = DQNAgent(agent0_params)
    model_path_0 = os.path.dirname(sys.path[0])+"/saved_models/double_escape/dqn/"+train_params["datetime"]+"/agent_0/model.h5"
    agent_1 = DQNAgent(agent1_params)
    model_path_1 = os.path.dirname(sys.path[0])+"/saved_models/double_escape/dqn/"+train_params["datetime"]+"/agent_1/model.h5"
    assert os.path.dirname(os.path.dirname(model_path_0)) == os.path.dirname(os.path.dirname(model_path_1))
    # init misc params
    update_counter = 0
    ep_returns = []
    start_time = time.time()
    for ep in range(train_params["num_episodes"]):
        epsilon_0 = agent_0.epsilon_decay(ep, train_params["num_episodes"])
        epsilon_1 = agent_1.epsilon_decay(ep, train_params["num_episodes"])
        print("epsilon_0: {}, epsilon_1: {}".format(epsilon_0, epsilon_1))
        pose_buffer = double_utils.create_pose_buffer(train_params["num_episodes"])
        theta_0, theta_1 = random.uniform(-math.pi, math.pi), random.uniform(-math.pi, math.pi)
        obs, _ = env.reset(pose_buffer[ep], theta_0, theta_1)
        state_agt0 = double_utils.obs_to_state(obs, "all")
        state_agt1 = double_utils.obs_to_state(obs, "all") # state of agent0 and agent1 could be same if using "all" option, when converting obs
        done, ep_rewards = False, []
        for st in range(train_params["num_steps"]):
            agent0_acti = agent_0.epsilon_greedy(state_agt0)
            agent0_action = agent_0.actions[agent0_acti]
            agent1_acti = agent_1.epsilon_greedy(state_agt1)
            agent1_action = agent_1.actions[agent1_acti]
            obs, rew, done, info = env.step(agent0_action, agent1_action)
            next_state_agt0 = double_utils.obs_to_state(obs, "all")
            next_state_agt1 = double_utils.obs_to_state(obs, "all")
            # adjust reward based on bonus options
            rew, done = double_utils.adjust_reward(train_params, env)
            ep_rewards.append(rew)
            # rew, done = double_utils.adjust_reward(hyp_params, env, agent)
            print(
                bcolors.OKGREEN,
                "Episode: {}, Step: {}: \naction0: {}->{}, action0: {}->{}, agent_0 state: {}, agent_1 state: {}, reward/episodic_return: {}/{}, status: {}, succeeds: {}".format(
                    ep,
                    st,
                    agent0_acti,
                    agent0_action,
                    agent1_acti,
                    agent1_action,
                    next_state_agt0,
                    next_state_agt1,
                    rew,
                    sum(ep_rewards),
                    info["status"],
                    env.success_count
                ),
                bcolors.ENDC
            )
            # store transition
            if not info["status"] == "blew":
                agent_0.replay_memory.store((state_agt0, agent0_acti, rew, done, next_state_agt0))
                agent_1.replay_memory.store((state_agt1, agent1_acti, rew, done, next_state_agt1))
                print(bcolors.OKBLUE, "transition saved to memory", bcolors.ENDC)
            else:
                print(bcolors.FAIL, "model blew up, transition not saved", bcolors.ENDC)
            agent_0.train()
            agent_1.train()
            state_agt0 = next_state_agt0
            state_agt1 = next_state_agt1
            update_counter += 1
            if not update_counter % agent_0.update_step:
                agent_0.qnet_stable.set_weights(agent_0.qnet_active.get_weights())
                print(bcolors.BOLD, "agent_0 Q-net weights updated!", bcolors.ENDC)
            if not update_counter % agent_1.update_step:
                agent_1.qnet_stable.set_weights(agent_1.qnet_active.get_weights())
                print(bcolors.BOLD, "agent_1 Q-net weights updated!", bcolors.ENDC)
            if done:
                break
        ep_returns.append(sum(ep_rewards))
        agent_0.save_model(model_path_0)
        agent_1.save_model(model_path_1)
    # time training
    end_time = time.time()
    training_time = end_time - start_time
    env.reset()

    # plot episodic returns
    data_utils.plot_returns(returns=ep_returns, mode=0, save_flag=True, fdir=os.path.dirname(os.path.dirname(model_path_0)))
    # plot accumulated returns
    data_utils.plot_returns(returns=ep_returns, mode=1, save_flag=True, fdir=os.path.dirname(os.path.dirname(model_path_0)))
    # plot averaged return
    data_utils.plot_returns(returns=ep_returns, mode=2, save_flag=True,
    fdir=os.path.dirname(os.path.dirname(model_path_0)))
    # save agent parameters
    data_utils.save_pkl(content=agent0_params, fdir=os.path.dirname(model_path_0), fname="agent0_parameters.pkl")
    data_utils.save_pkl(content=agent1_params, fdir=os.path.dirname(model_path_1), fname="agent1_parameters.pkl")
    # save returns
    data_utils.save_pkl(content=ep_returns, fdir=os.path.dirname(os.path.dirname(model_path_0)), fname="episodic_returns.pkl")
    # save results
    train_info = train_params
    train_info["success_count"] = env.success_count
    train_info["training_time"] = training_time
    train_info["agent0_learning_rate"] = agent0_params["learning_rate"]
    train_info["agent0_state_dimension"] = agent0_params["dim_state"]
    train_info["agent0_action_options"] = agent0_params["actions"]
    train_info["agent0_layer_sizes"] = agent0_params["layer_sizes"]
    train_info["agent1_learning_rate"] = agent1_params["learning_rate"]
    train_info["agent1_state_dimension"] = agent1_params["dim_state"]
    train_info["agent1_action_options"] = agent1_params["actions"]
    train_info["agent1_layer_sizes"] = agent1_params["layer_sizes"]
    data_utils.save_csv(content=train_info, fdir=os.path.dirname(os.path.dirname(model_path_0)), fname="train_information.csv")
