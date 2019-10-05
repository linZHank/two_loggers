#! /usr/bin/env python
"""
An implementation of Deep Q-network (DQN) for solo_escape_task
DQN is a Model free, off policy, reinforcement learning algorithm (https://deepmind.com/research/dqn/)
Author: LinZHanK (linzhank@gmail.com)
"""
from __future__ import absolute_import, division, print_function

import sys
import os
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
import rospy

from envs.solo_escape_task_env import SoloEscapeEnv
from utils import data_utils, solo_utils, tf_utils
from utils.data_utils import bcolors
from agents.dqn import DQNAgent

if __name__ == "__main__":
    # args = solo_utils.get_args()
    # make an instance from env class
    env = SoloEscapeEnv()
    env.reset()
    # create training parameters
    date_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    if not args.source: # source is empty, create new params
        complete_episodes = 0
        train_params = solo_utils.create_train_params(date_time, complete_episodes, args.source, args.normalize, args.num_episodes, args.num_steps, args.time_bonus, args.wall_bonus, args.door_bonus, args.success_bonus)
        # init agent parameters
        dim_state = len(double_utils.obs_to_state(env.observation, "all"))
        action = np.zeros(2)
        layer_sizes = [128]
        gamma = 0.99
        learning_rate = 3e-4
        batch_size = 2048
        memory_cap = 100000
        update_step = 8192
        decay_period = args.num_episodes/4
        final_eps = 1e-2
        agent_params = solo_utils.create_agent_params(dim_state, actions, layer_sizes, gamma, learning_rate, batch_size, memory_cap, update_step, decay_period, final_eps)
        agent_params['update_counter'] = 0
        # instantiate new agents
        agent = DQNAgent(agent_params)
        model_path = os.path.dirname(sys.path[0])+"/saved_models/solo_escape/dqn/"+date_time+"/agent/model.h5"
        # init returns and losses storage
        train_params['ep_returns'] = []
        agent_params['ep_losses'] = []
        # init random starting poses
        train_params['pose_buffer'] = []
        # init first episode and step
        obs, _ = env.reset()
        # train_params['pose_buffer'].append()
        state= double_utils.obs_to_state(obs, "all")
        train_params['success_count'] = 0
        # new means and stds
        mean_0 = state_0 # states average
        std_0 = np.zeros(agent_params_0["dim_state"])+1e-8 # n*Var
        mean_1 = state_1 # states average
        std_1 = np.zeros(agent_params_1["dim_state"])+1e-8 # n*Var
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
        # init robots from loaded pose buffer
        obs, _ = env.reset(train_params['pose_buffer'][train_params['complete_episodes']])
        state_0 = double_utils.obs_to_state(obs, "all")
        state_1 = double_utils.obs_to_state(obs, "all")
        env.success_count = train_params['success_count']
        # load means and stds
        mean_0 = agent_params_0['mean']
        std_0 = agent_params_0['std']
        mean_1 = agent_params_1['mean']
        std_1 = agent_params_1['std']

    agent_params = {}
    train_params = {}
    # agent parameters
    agent_params["dim_state"] = len(solo_utils.obs_to_state(env.observation))
    agent_params["actions"] = np.array([np.array([.5, -1]), np.array([.5, 1])])
    agent_params["layer_size"] = [64,64]
    agent_params["gamma"] = 0.99
    agent_params["learning_rate"] = 3e-4
    agent_params["batch_size"] = 2000
    agent_params["memory_cap"] = 500000
    agent_params["update_step"] = 10000
    agent_params["model_path"] = os.path.dirname(sys.path[0])+"/saved_models/solo_escape/dqn_model/"+datetime.now().strftime("%Y-%m-%d-%H-%M")+"/model.ckpt"
    # training params
    train_params["num_episodes"] = 6000
    train_params["num_steps"] = 256
    train_params["time_bonus"] = True
    train_params["dist_bonus"] = False
    train_params["success_bonus"] = 10
    train_params["wall_bonus"] = -1./100
    train_params["door_bonus"] = 1./100
    # instantiate agent
    agent = DQNAgent(agent_params)
    update_counter = 0
    ep_returns = []
    for ep in range(train_params["num_episodes"]):
        epsilon = agent.epsilon_decay(ep, train_params["num_episodes"])
        print("epsilon: {}".format(epsilon))
        obs, _ = env.reset()
        state_0 = solo_utils.obs_to_state(obs)
        dist_0 = np.linalg.norm(state_0[:2]-np.array([0,-6.0001]))
        done, ep_rewards = False, []
        for st in range(train_params["num_steps"]):
            act_id = agent.epsilon_greedy(state_0)
            action = agent.actions[act_id]
            obs, rew, done, info = env.step(action)
            state_1 = solo_utils.obs_to_state(obs)
            dist_1 = np.linalg.norm(state_1[:2]-np.array([0,-6.0001]))
            agent.delta_dist = dist_0 - dist_1
            # adjust reward based on bonus options
            rew, done = solo_utils.adjust_reward(train_params, env, agent)
            print(
                bcolors.OKGREEN,
                "Episode: {}, Step: {} \naction: {}->{}, state: {}, reward: {}, status: {}".format(
                    ep,
                    st,
                    act_id,
                    action,
                    state_1,
                    rew,
                    info
                ),
                bcolors.ENDC
            )
            # store transition
            agent.replay_memory.store((state_0, act_id, rew, done, state_1))
            agent.train()
            state_0 = state_1
            ep_rewards.append(rew)
            update_counter += 1
            if not update_counter % agent.update_step:
                agent.qnet_stable.set_weights(agent.qnet_active.get_weights())
                print(bcolors.BOLD, "Q-net weights updated!", bcolors.ENDC)
            if done:
                break
        ep_returns.append(sum(ep_rewards))
        print(bcolors.OKBLUE, "Episode: {}, Success Count: {}".format(ep, env.success_count),bcolors.ENDC)
        agent.save_model()
        print("model saved at {}".format(agent.model_path))
    # plot deposit returns
    data_utils.plot_returns(returns=ep_returns, mode=2, save_flag=True, path=agent_params["model_path"])

    data_utils.save_pkl(content=agent_params, path=agent_params["model_path"], fname="agent_parameters.pkl")
    # save results
    train_info = train_params.update(agent_params)
    train_info["success_count"] = env.success_count
    data_utils.save_csv(content=train_info, path=agent_params["model_path"], fname="train_information.csv")
