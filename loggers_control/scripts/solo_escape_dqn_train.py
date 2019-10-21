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
import pickle

from envs.solo_escape_task_env import SoloEscapeEnv
from utils import data_utils, solo_utils, tf_utils
from utils.data_utils import bcolors
from agents.dqn import DQNAgent
from agents import dqn

import pdb

if __name__ == "__main__":
    args = solo_utils.get_args()
    # make an instance from env class
    env = SoloEscapeEnv()
    env.unpauseSim()
    env.reset()

    # new training or continue training
    date_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    model_path = sys.path[0]+"/saved_models/solo_escape/dqn/"+date_time+"/agent/model.h5"
    if not args.source: # source is empty, create new params
        complete_episodes = 0
        train_params = solo_utils.create_train_params(date_time, complete_episodes, args.source, args.normalize, args.num_episodes, args.num_steps, args.time_bonus, args.wall_bonus, args.door_bonus, args.success_bonus)
        # agent parameters
        dim_state = len(solo_utils.obs_to_state(env.observation))
        actions = np.array([np.array([1, -1]), np.array([1, 1])])
        layer_sizes = [128, 64]
        gamma = 0.99
        learning_rate = 1e-3
        batch_size = 2048
        memory_cap = 100000
        update_step = 8192
        decay_period = train_params['num_episodes']/3
        init_eps = 1.
        final_eps = 1e-2
        agent_params = dqn.create_agent_params(dim_state, actions, layer_sizes, gamma, learning_rate, batch_size, memory_cap, update_step, decay_period, init_eps, final_eps)
        agent_params['update_counter'] = 0
        # instantiate new agents
        agent = DQNAgent(agent_params)
        # init returns and losses storage
        ep_returns = []
        ep_losses = []
        # init first episode and step
        obs, _ = env.reset()
        state = solo_utils.obs_to_state(obs)
        train_params['success_count'] = 0
        # new means and stds
        mean = state # states average
        std = np.zeros(agent_params["dim_state"])+1e-15 # n*Var
    else: # source is not empty, load params
        model_load_dir = os.path.dirname(sys.path[0])+"/saved_models/solo_escape/dqn/"+args.source
        # load train parameters
        train_params_path = os.path.join(model_load_dir, "train_parameters.pkl")
        with open(train_params_path, 'rb') as f:
            train_params = pickle.load(f)
        train_params['source'] = args.source
        train_params["date_time"] = date_time
        # load agents parameters
        agent_params_path = os.path.join(model_load_dir,"agent/agent_parameters.pkl")
        with open(agent_params_path, 'rb') as f:
            agent_params = pickle.load(f) # load agent_0 model
        if args.num_episodes > train_params['num_episodes']: # continue from an ended training, else, continue from a crashed training
            train_params['num_episodes'] = args.num_episodes
            agent_params['init_eps'] = 0.5
        agent_params['decay_period'] = train_params['num_episodes']/3
        # load dqn models & memory buffers
        agent = DQNAgent(agent_params)
        agent.load_model(os.path.join(model_load_dir, "agent/model.h5"))
        ep_returns = np.load(os.path.join(model_load_dir, 'agent/ep_returns.npy')).tolist()
        ep_losses = np.load(os.path.join(model_load_dir, 'agent/ep_losses.npy')).tolist()
        # initialize robot from loaded pose buffer
        obs, _ = env.reset()
        state = solo_utils.obs_to_state(obs)
        env.success_count = train_params['success_count']
        # load means and stds
        mean = agent_params['mean']
        std = agent_params['std']

    # learning
    start_time = time.time()
    for ep in range(train_params['complete_episodes'], train_params['num_episodes']):
        # check simulation crash
        if sum(np.isnan(state)):
            rospy.logfatal("Simulation Crashed")
            train_params['complete_episodes'] = ep
            break # terminate main loop if simulation crashed
        epsilon = agent.linearly_decaying_epsilon(decay_period=agent_params['decay_period'], episode=ep, init_eps=agent_params['init_eps'], final_eps=agent_params['final_eps'])
        rospy.logdebug("epsilon: {}".format(epsilon))
        done, ep_rewards, loss_vals = False, [], []
        for st in range(train_params['num_steps']):
            # check simulation crash
            if sum(np.isnan(state)):
                rospy.logfatal("Simulation Crashed")
                break # terminate main loop if simulation crashed
            # normalize states
            if train_params['normalize']:
                norm_state = tf_utils.normalize(state, mean, std)
                rospy.logdebug("State normalized from {} \nto {}".format(state, norm_state))
            else:
                norm_state = state
            action_index = agent.epsilon_greedy(norm_state)
            action = agent.actions[action_index]
            # take an action
            obs, rew, done, info = env.step(action)
            next_state = solo_utils.obs_to_state(obs)
            # compute incremental mean and std
            inc_mean = tf_utils.increment_mean(mean, next_state, (ep+1)*(st+1)+1)
            inc_std = tf_utils.increment_std(std, mean, inc_mean, next_state, (ep+1)*(st+1)+1)
            # update mean and std
            agent_params['mean'] = mean
            agent_params['std'] = std
            # normalize next state
            if train_params['normalize']:
                norm_next_state = tf_utils.normalize(next_state, mean, std)
                rospy.logdebug("Next states normalized from {} \nto {}".format((next_state, norm_next_state)))
            else:
                norm_next_state = next_state
            # adjust reward based on bonus args
            rew, done = solo_utils.adjust_reward(train_params, env)
            ep_rewards.append(rew)
            # store transitions
            if not info["status"] == "blew":
                agent.replay_memory.store((norm_state, action_index, rew, done, norm_next_state))
                print(bcolors.OKBLUE, "transition saved to memory", bcolors.ENDC)
            else:
                print(bcolors.FAIL, "model blew up, transition not saved", bcolors.ENDC)
            # env.pauseSim()
            agent.train()
            # env.unpauseSim()
            loss_vals.append(agent.loss_value)
            state = next_state
            agent_params['update_counter'] += 1
            # log step
            rospy.loginfo(
                "Episode: {}, Step: {}, epsilon: {} \nstate: {}, action: {}, next state: {} \nreward/episodic_return: {}/{}, status: {}, number of success: {}".format(
                    ep+1,
                    st+1,
                    agent.epsilon,
                    state,
                    action,
                    next_state,
                    rew,
                    sum(ep_rewards),
                    info["status"],
                    env.success_count
                )
            )
            if not agent_params['update_counter'] % agent_params['update_step']:
                agent.qnet_stable.set_weights(agent.qnet_active.get_weights())
                rospy.loginfo("agent Q-net weights updated!")
            if done:
                train_params['complete_episodes'] += 1
                rospy.logwarn(
                    "Episode {} summary \n---\ntotal steps: {}, training consumed: {} seconds".format(ep+1, st+1, time.time()-start_time)
                )
                break
        ep_returns.append(sum(ep_rewards))
        ep_losses.append(sum(loss_vals)/len(loss_vals))
        agent.save_model(model_path)
        obs, _ = env.reset()
        state = solo_utils.obs_to_state(obs)
    # time
    env.pauseSim() # check sim time in gazebo window

    # save replay buffer memories
    agent.save_memory(model_path)
    # save agent parameters
    data_utils.save_pkl(content=agent_params, fdir=os.path.dirname(model_path), fname="agent_parameters.pkl")
    data_utils.save_pkl(content=train_params, fdir=os.path.dirname(os.path.dirname(model_path)), fname="train_parameters.pkl")
    # save train info
    train_info = train_params
    train_info['success_count'] = env.success_count
    train_info['train_dur'] = time.time() - start_time
    train_info["agent_learning_rate"] = agent_params["learning_rate"]
    train_info["agent_state_dimension"] = agent_params["dim_state"]
    train_info["agent_action_options"] = agent_params["actions"]
    train_info["agent_layer_sizes"] = agent_params["layer_sizes"]
    data_utils.save_csv(content=train_info, fdir=os.path.dirname(os.path.dirname(model_path)), fname="train_information.csv")
    # save results
    np.save(os.path.join(os.path.dirname(model_path), 'ep_returns.npy'), ep_returns)
    np.save(os.path.join(os.path.dirname(model_path), 'ep_losses.npy'), ep_losses)

    # plot episodic returns
    data_utils.plot_returns(returns=ep_returns, mode=0, save_flag=True, fdir=os.path.dirname(os.path.dirname(model_path)))
    # plot accumulated returns
    data_utils.plot_returns(returns=ep_returns, mode=1, save_flag=True, fdir=os.path.dirname(os.path.dirname(model_path)))
    # plot averaged return
    data_utils.plot_returns(returns=ep_returns, mode=2, save_flag=True,
    fdir=os.path.dirname(os.path.dirname(model_path)))
