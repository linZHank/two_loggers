#! /usr/bin/env python
"""
An implementation of Vanilla Policy Gradient (VPG) for solo_escape_task
VPG is a model free, on policy, reinforcement learning algorithm (https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)
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
from utils import gen_utils, solo_utils, tf_utils
from utils.gen_utils import bcolors
from agents.vpg import VPGAgent

if __name__ == "__main__":
    # create argument parser
    args = gen_utils.get_args()
    # start timing training
    start_time = time.time()
    rospy.init_node("solo_escape_dqn", anonymous=True, log_level=rospy.INFO)
    # make an instance from env class
    env = SoloEscapeEnv()
    env.reset()
    agent_params = {}
    train_params = {}
    # agent parameters
    agent_params["dim_state"] = len(solo_utils.obs_to_state(env.observation))
    agent_params["actions"] = np.array([np.array([1, -1]), np.array([1, 1])])
    agent_params["layer_sizes"] = args.layer_sizes
    agent_params["learning_rate"] = args.learning_rate
    # training params
    if args.datetime:
        train_params["datetime"] = args.datetime
    else:
        train_params["datetime"] = datetime.now().strftime("%Y-%m-%d-%H-%M")
    train_params["num_epochs"] = args.num_epochs
    train_params["num_steps"] = args.num_steps
    train_params["time_bonus"] = -1./train_params['num_steps']
    train_params["success_bonus"] = 0
    train_params["wall_bonus"] = -1./100
    train_params["door_bonus"] = 0
    train_params["sample_size"] = args.sample_size
    # instantiate agent
    agent = VPGAgent(agent_params)
    update_counter = 0
    episodic_returns = []
    episode = 0
    step = 0
    for ep in range(train_params['num_epochs']):
        # init training batches
        batch_states = []
        batch_acts = []
        batch_rtaus = []
        # init episode
        obs, _ = env.reset()
        state_0 = solo_utils.obs_to_state(obs)
        done, ep_rewards = False, []
        batch_counter = 0
        while True:
            # take action by sampling policy_net predictions
            act_id = agent.sample_action(state_0)
            action = agent.actions[act_id]
            obs, rew, done, info = env.step(action)
            state_1 = solo_utils.obs_to_state(obs)
            # adjust reward
            rew, done = solo_utils.adjust_reward(train_params, env)
            # fill training batch
            batch_acts.append(act_id)
            batch_states.append(state_0)
            # update
            ep_rewards.append(rew)
            state_0 = state_1
            print(
                bcolors.OKGREEN,
                "Epoch: {} \nEpisode: {}, Step: {} \naction: {}->{}, state: {}, reward/episodic_return: {}/{}, status: {}, success: {}".format(
                    ep,
                    episode,
                    step,
                    act_id,
                    action,
                    state_1,
                    rew,
                    sum(ep_rewards),
                    info,
                    env.success_count
                ),
                bcolors.ENDC
            )
            # step increment
            step += 1
            if done:
                ep_return, ep_length = sum(ep_rewards), len(ep_rewards)
                batch_rtaus += list(solo_utils.reward_to_go(ep_rewards))
                assert len(batch_rtaus) == len(batch_states)
                # store episodic_return
                episodic_returns.append(ep_return)
                # reset to a new episode
                obs, _ = env.reset()
                done, ep_rewards = False, []
                state_0 = solo_utils.obs_to_state(obs)
                episode += 1
                step = 0
                print(
                    bcolors.OKGREEN,
                    "current batch size: {}".format(len(batch_rtaus)),
                    bcolors.ENDC
                )
                if len(batch_rtaus) > train_params['sample_size']:
                    break
        agent.train(batch_states, batch_acts, batch_rtaus)
        # specify model path
        model_path = os.path.dirname(sys.path[0])+"/saved_models/solo_escape/vpg/"+train_params["datetime"]+"/agent/model.h5"
        agent.save_model(model_path)
    # time training
    end_time = time.time()
    training_time = end_time - start_time

    # plot episodic returns
    gen_utils.plot_returns(returns=episodic_returns, mode=0, save_flag=True, path=os.path.dirname(model_path))
    # plot accumulated returns
    gen_utils.plot_returns(returns=episodic_returns, mode=1, save_flag=True, path=os.path.dirname(model_path))
    # plot averaged return
    gen_utils.plot_returns(returns=episodic_returns, mode=2, save_flag=True,
    path=os.path.dirname(model_path))
    # save returns
    gen_utils.save_pkl(content=episodic_returns, path=os.path.dirname(model_path), fname="episodic_returns.pkl")
    # save agent parameters
    gen_utils.save_pkl(content=agent_params, path=os.path.dirname(model_path), fname="agent_parameters.pkl")
    # save results
    train_info = train_params
    train_info["success_count"] = env.success_count
    train_info["training_time"] = training_time
    train_info["learning_rate"] = agent_params["learning_rate"]
    train_info["state_dimension"] = agent_params["dim_state"]
    train_info["action_options"] = agent_params["actions"]
    train_info["layer_sizes"] = agent_params["layer_sizes"]
    gen_utils.save_csv(content=train_info, path=os.path.dirname(model_path), fname="train_information.csv")
