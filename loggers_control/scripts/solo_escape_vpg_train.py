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
    agent_params["batch_size"] = 2000
    agent_params["model_path"] = os.path.dirname(sys.path[0])+"/saved_models/solo_escape/dqn_model/"+datetime.now().strftime("%Y-%m-%d-%H-%M")+"/model.ckpt"
    # training params
    train_params["num_episodes"] = 6000
    train_params["num_steps"] = 256
    train_params["time_bonus"] = True
    train_params["success_bonus"] = 10
    train_params["wall_bonus"] = -1./100
    train_params["door_bonus"] = 0
    # instantiate agent
    agent = VPGAgent(agent_params)
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
    gen_utils.plot_returns(returns=ep_returns, mode=2, save_flag=True, path=agent_params["model_path"])

    gen_utils.save_pkl(content=agent_params, path=agent_params["model_path"], fname="agent_parameters.pkl")
    # save results
    train_info = train_params.update(agent_params)
    train_info["success_count"] = env.success_count
    gen_utils.save_csv(content=train_info, path=agent_params["model_path"], fname="train_information.csv")
