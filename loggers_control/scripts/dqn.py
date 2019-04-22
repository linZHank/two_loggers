from __future__ import absolute_import, division, print_function

import sys
import os
import numpy as np
import tensorflow as tf
import envs
import utils

# class DQNAgent:
#     def __init__(self):
#         pass
#
#     def test(self, env):
#         obs, _ = env.reset()
#         done, ep_rewards = False, []
#         state = env_utils.obs_to_state(obs)
#         while not done:
#              action = self.qnet_model.action_value(state)
#              obs, rew, done, info = env.step(action)
#              ep_rewards.append(rew)
#              state = env_utils.obs_to_state(obs)
#         return ep_rewards
#
#
#
#
#     def train(self, env):
#         pass
