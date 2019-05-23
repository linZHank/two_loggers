#! /usr/bin/env python
"""
An implementation of Vanilla Policy Gradient (VPG) for solo_escape_task
VPG is a model free, on policy, reinforcement learning algorithm (https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)
Author: LinZHanK (linzhank@gmail.com)
"""
from __future__ import absolute_import, division, print_function

import sys
import os
from datetime import datetime
import numpy as np
import tensorflow as tf
import rospy

from envs.solo_escape_task_env import SoloEscapeEnv
from utils import gen_utils, solo_utils, tf_utils
from utils.gen_utils import bcolors
from agents.vpg import VPGAgent

if __name__ == "__main__":
    # start_time = time.time()
    rospy.init_node("solo_escape_dqn", anonymous=True, log_level=rospy.INFO)
    # make an instance from env class
    env = SoloEscapeEnv()
    env.reset()
    agent_params = {}
    train_params = {}
    # agent parameters
    agent_params["dim_state"] = len(solo_utils.obs_to_state(env.observation))
    agent_params["actions"] = np.array([np.array([.5, -1]), np.array([.5, 1])])
    agent_params["layer_size"] = [64,64]
    agent_params["gamma"] = 0.99
    agent_params["learning_rate"] = 3e-4
    agent_params["batch_size"] = 1024
    agent_params["model_path"] = os.path.dirname(sys.path[0])+"/saved_models/solo_escape/vpg_model/"+datetime.now().strftime("%Y-%m-%d-%H-%M")+"/agent/model.h5"
    # training params
    train_params["num_epochs"] = 1000
    train_params["num_steps"] = 256
    train_params["time_bonus"] = -1./train_params['num_steps']
    train_params["success_bonus"] = 0
    train_params["wall_bonus"] = -1./100
    train_params["door_bonus"] = 0
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
            rew, done = solo_utils.adjust_reward(train_params, env, agent)
            # fill training batch
            batch_acts.append(act_id)
            batch_states.append(state_0)
            # update
            ep_rewards.append(rew)
            state_0 = state_1
            print(
                bcolors.OKGREEN,
                "Episode: {}, Step: {} \naction: {}->{}, state: {}, reward/episodic_return: {}/{}, status: {}, success: {}".format(
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
                batch_rtaus += [ep_return] * ep_length
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
                if len(batch_rtaus) > agent_params['batch_size']:
                    break
        agent.train(batch_states, batch_acts, batch_rtaus)
        agent.save_model()
    # plot deposit returns
    gen_utils.plot_returns(returns=episodic_returns, mode=2, save_flag=True, path=os.path.dirname(agent_params["model_path"]))
    # save results
    gen_utils.save_pkl(content=agent_params, path=agent_params["model_path"], fname="agent_parameters.pkl")
    # save results
    train_info = train_params
    train_info["success_count"] = env.success_count
    train_info["agent"] = agent_params
    gen_utils.save_csv(content=train_info, path=os.path.dirname(agent_params["model_path"]), fname="train_information.csv")
