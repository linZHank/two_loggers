from __future__ import absolute_import, division, print_function

import numpy as np
import ag_utils
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model


class Memory:
    def __init__(self, memory_cap):
        self.memory_cap = memory_cap
        self.memory = []
    def store(self, experience):
        # pop a random experience if memory full
        if len(self.memory) >= self.memory_cap:
            self.memory.pop(np.random.randint(len(self.memory))
        self.memory.append(experience)
    def sample_batch(self, batch_size):
        # Select batch
        if len(self.memory) < batch_size:
            batch = random.sample(self.memory, len(self.memory))
        else:
            batch = random.sample(self.memory, batch_size)

        return zip(*batch)

class DQNAgent:
    def __init__(self):
        pass

    def train(self, env, actions, dim_state, num_actions,
              batch_size=1000, num_episodes=512, num_steps=1000):
        replay_memory = Memory(memory_cap=50000) # init replay memory
        qnet_active = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, input_shape=(7, ), activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(num_actions, activation='softmax')
        ]) # init Q(s,a)
        qnet_stable = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, input_shape=(7, ), activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(num_actions, activation='softmax')
        ]) # init Q^(s,a)
        states = np.empty((batch_size, dim_state))
        actions = np.empty((batch_size,), dtype=np.int32)
        update_counter = 10000
        for ep in range(num_episodes):
            obs, _ = env.reset()
            state_0 = utils.obs_to_state(obs)
            done, ep_rewards = False, []
            for st in range(num_steps):
                action =  utils.epsilon_greedy(qnet_active, state_0, epsilon)
                obs, rew, done, info = env.step(action)
                state_1 = utils.obs_to_state(obs)
                replay_memory.store((state_0, action, rew, done, state_1))
                sample = replay_memory.sample_batch(batch_size)
                # create dataset
                utils.set_q_target(utils.sample_batch(replay_memory, batch_size), self.qnet_hat)
                utils.train_batch
                if update_weights:
                    self.qnet_hat = self.qnet
                if done:
                    break
