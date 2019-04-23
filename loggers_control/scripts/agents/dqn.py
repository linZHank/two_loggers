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
    def __init__(self, hyp_params):
        super(DQNAgent, self).__init__()
        # hyper-parameters
        self.epsilon = hyp_params["epsilon"]
        self.actions = hyp_params["actions"]
        self.gamma = hyp_params["gamma"]
        # Q(s,a;theta)
        self.qnet_active = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, input_shape=(dim_state, ), activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(num_actions, activation='softmax')
        ])
        self.qnet_active.compile(optimizer="adam",
                            loss="sparse_categorical_crossentropy",
                            metrics=["accuracy"])
        # Q^(s,a;theta_)
        self.qnet_stable = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, input_shape=(dim_state, ), activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(num_actions, activation='softmax')
        ])
        # init replay memory
        self.replay_memory = Memory(memory_cap=50000)

    def epsilon_greedy(self, state):
        """
        If a random number(0,1) beats epsilon, return index of largest Q-value.
        Else, return a random index
        """
        if np.random.rand() > self.epsilon:
            return np.argmax(self.qnet_active.predict(state.reshape(1,-1))
        else:
            return np.random.randint(len(self.actions))

    def epsilon_decay(self, epi):
        return 1./(epi + 20)

    def compute_target_q(self, sampled_batch):
        target_q = np.array(sampled_batch[1]) + self.gamma *  np.max(self.qnet_stable.predict(np.array(sampled_batch[-1])), axis=1)



    def train(self, env):
        update_step = 1000
        total_step = 0
        for ep in range(num_episodes):
            self.epsilon = self.epsilon_decay(ep)
            obs, _ = env.reset()
            state_0 = ag_utils.obs_to_state(obs)
            done, ep_rewards = False, []
            for st in range(num_steps):
                act_id = self.epsilon_greedy(state_0)
                action = actions[act_id]
                obs, rew, done, info = env.step(action)
                state_1 = ag_utils.obs_to_state(obs)
                replay_memory.store((state_0, act_id, rew, done, state_1))
                minibatch = replay_memory.sample_batch(batch_size)
                # create dataset
                x = np.array(minibatch[0])
                y = self.compute_target_q(minibatch)
                total_step += 1
                if total_step % update_step == 0:
                    self.qnet_stable.set_weights(qnet_active.get_weights())
                if done:
                    break
