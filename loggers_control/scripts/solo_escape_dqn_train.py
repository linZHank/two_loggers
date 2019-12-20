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
from utils import data_utils, solo_utils
from agents.dqn import DQNAgent
from agents import dqn

import pdb

if __name__ == "__main__":
    args = solo_utils.get_args()
    # make an instance from env class
    env = SoloEscapeEnv()
    env.reset()
    # new training or continue training
    if not args.source: # source is empty, create new params
        rospy.logwarn("Start a new training")
        date_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
        dim_state = len(solo_utils.obs_to_state(env.observation))
        actions = np.array([np.array([1, -1]), np.array([1, 1])])
        # train parameters
        train_params = solo_utils.create_train_params(complete_episodes=0, complete_steps=0, success_count=0, source=args.source, normalize=args.normalize, num_episodes=args.num_episodes, num_steps=args.num_steps, time_bonus=args.time_bonus, wall_bonus=args.wall_bonus, door_bonus=args.door_bonus, success_bonus=args.success_bonus)
        # agent parameters
        agent_params = solo_utils.create_agent_params(dim_state=dim_state, actions=actions, ep_returns=[], ep_losses=[], mean=np.zeros(dim_state), std=np.zeros(dim_state)+1e-15, layer_sizes=args.layer_sizes, discount_rate=args.gamma, learning_rate=args.lr, batch_size=args.batch_size, memory_cap=args.mem_cap, update_step=args.update_step, decay_period=train_params['num_episodes']*9/10, init_eps=args.init_eps, final_eps=args.final_eps)
        # instantiate new agents
        agent = DQNAgent(agent_params)
        model_path = os.path.dirname(sys.path[0])+"/saved_models/solo_escape/dqn/"+date_time+"/agent_0/model.h5"
    else: # source is not empty, load params
        rospy.logwarn("Continue training from source: {}".format(args.source))
        # load train parameters
        model_path = os.path.dirname(sys.path[0])+"/saved_models/solo_escape/dqn/"+args.source+"/agent/model.h5"
        with open(os.path.dirname(os.path.dirname(model_path))+ "/train_parameters.pkl", 'rb') as f:
            train_params = pickle.load(f)
        # load agents parameters
        with open(os.path.dirname(model_path)+'/agent_parameters.pkl', 'rb') as f:
            agent_params = pickle.load(f) # load agent_0 model
        # load dqn models & memory buffers
        agent = DQNAgent(agent_params)
        agent.load_model(model_path)

    # learning
    start_time = time.time()
    mean = agent_params['mean']
    std = agent_params['std']
    for ep in range(train_params['complete_episodes'], train_params['num_episodes']):
        obs, info = env.reset()
        state = solo_utils.obs_to_state(obs)
        # check simulation crash
        if sum(np.isnan(state)):
            rospy.logfatal("Simulation Crashed")
            train_params['complete_episodes'] = ep
            break # terminate main loop if simulation crashed
        if info["status"] == "blew":
            rospy.logerr("Model blew up, skip this episode")
            continue
        epsilon = agent.linearly_decaying_epsilon(episode=ep)
        done, ep_rewards, loss_vals = False, [], []
        for st in range(train_params['num_steps']):
            # check simulation crash
            if sum(np.isnan(state)):
                rospy.logfatal("Simulation Crashed")
                break # terminate main loop if simulation crashed
            # normalize states
            if train_params['normalize']:
                refine_state = data_utils.normalize(state, mean, std)
                rospy.logdebug("State normalized from {} \nto {}".format(state, refine_state))
            else:
                refine_state = state
            action_index = agent.epsilon_greedy(refine_state)
            action = agent.actions[action_index]
            # take an action
            obs, rew, done, info = env.step(action)
            next_state = solo_utils.obs_to_state(obs)
            # compute incremental mean and std
            inc_mean = data_utils.increment_mean(mean, next_state, (ep+1)*(st+1)+1)
            inc_std = data_utils.increment_std(std, mean, inc_mean, next_state, (ep+1)*(st+1)+1)
            # update mean and std
            mean, std = inc_mean, inc_std
            agent_params['mean'] = mean
            agent_params['std'] = std
            # normalize next state
            if train_params['normalize']:
                refine_next_state = data_utils.normalize(next_state, mean, std)
                rospy.logdebug("Next states normalized from {} \nto {}".format((next_state, refine_next_state)))
            else:
                refine_next_state = next_state
            # adjust reward based on bonus args
            rew, done = solo_utils.adjust_reward(train_params, env)
            ep_rewards.append(rew)
            train_params['complete_steps'] += 1
            # store transitions
            if not info["status"] == "blew":
                agent.replay_memory.store((refine_state, action_index, rew, done, refine_next_state))
                rospy.logwarn("transition saved to memory")
            else:
                rospy.logerr("model blew up, transition not saved")
            # log step
            rospy.loginfo(
                "Episode: {}, Step: {}, epsilon: {} \nstate: {}, action: {}, next state: {} \nreward/episodic_return: {}/{}, status: {}, number of success: {}".format(
                    ep+1,
                    st+1,
                    agent.epsilon,
                    refine_state,
                    action,
                    refine_next_state,
                    rew,
                    sum(ep_rewards),
                    info["status"],
                    env.success_count
                )
            )
            # train one epoch
            agent.train()
            loss_vals.append(agent.loss_value)
            state = next_state
            # update q-statble net
            if not train_params['complete_steps'] % agent_params['update_step']:
                agent.qnet_stable.set_weights(agent.qnet_active.get_weights())
                rospy.logerr("agent Q-net weights updated!")
            if done:
                rospy.logwarn(
                    "Episode {} summary \n---\ntotal steps: {}, episode consumed: {} seconds".format(ep+1, st+1, time.time()-start_time)
                )
                break
        train_params['success_count'] += env.success_count
        train_params['complete_episodes'] += 1
        agent_params['ep_returns'].append(sum(ep_rewards))
        agent_params['ep_losses'].append(sum(loss_vals)/len(loss_vals))
        agent.save_model(model_path)

    # time
    end_time = time.time()
    train_dur = end_time - start_time
    env.reset()

    # save replay buffer memories
    agent.save_memory(model_path)
    # save agent parameters
    data_utils.save_pkl(content=agent_params, fdir=os.path.dirname(model_path), fname="agent_parameters.pkl")
    # create train info
    train_info = train_params
    train_info["train_dur"] = train_dur
    train_info["agent_learning_rate"] = agent_params["learning_rate"]
    train_info["agent_state_dimension"] = agent_params["dim_state"]
    train_info["agent_action_options"] = agent_params["actions"]
    train_info["agent_layer_sizes"] = agent_params["layer_sizes"]
    # save train info
    data_utils.save_csv(content=train_info, fdir=os.path.dirname(os.path.dirname(model_path)), fname="train_information.csv")
    data_utils.save_pkl(content=train_params, fdir=os.path.dirname(os.path.dirname(model_path)), fname="train_parameters.pkl")
    # save results
    np.save(os.path.join(os.path.dirname(model_path), 'ep_returns.npy'), agent_params['ep_returns'])
    np.save(os.path.join(os.path.dirname(model_path), 'ep_losses.npy'), agent_params['ep_losses'])

    # plot episodic returns
    data_utils.plot_returns(returns=agent_params['ep_returns'], mode=0, save_flag=True, fdir=os.path.dirname(os.path.dirname(model_path)))
    # plot accumulated returns
    data_utils.plot_returns(returns=agent_params['ep_returns'], mode=1, save_flag=True, fdir=os.path.dirname(os.path.dirname(model_path)))
    # plot averaged return
    data_utils.plot_returns(returns=agent_params['ep_returns'], mode=2, save_flag=True,
    fdir=os.path.dirname(os.path.dirname(model_path)))
