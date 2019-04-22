from __future__ import absolute_import, division, print_function

import ag_utils
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

class QNet(Model):
    def __init__(self, num_actions):
        super(Model,self).__init__()
        self.hidden1 = Dense(128, activation="relu")
        self.hidden2 = Dense(128, activation="relu")
        self.output= Dense(num_actions)

    def call(self, state):
        feature = self.hidden1(state)
        feature = self.hidden2(feature)
        return self.output(feature)

class DQNAgent:
    def __init__(self):
        pass

    def train(self, env, actions, dim_state, num_actions,
              batch_size=1000, num_episodes=512, num_steps=1000):
        states = mp.empty((batch_size, dim_state))
        actions = np.empty((batch_size,), dtype=np.int32)
        for ep in range(num_episodes):
            obs, _ = env.reset()
            state_0 = utils.obs_to_state(obs)
            done, ep_rewards = False, []
            for st in range(num_steps):
                action = utils.epsilon_greedy(self.qnet(state_0), epsilon)
                obs, rew, done, info = env.step(action)
                state_1 = utils.obs_to_state(obs)
                utils.store_experience(state_0, action, rew, done, state_1, replay_memory)
                utils.set_q_target(utils.sample_batch(replay_memory, batch_size), self.qnet_hat)
                utils.train_batch
                if update_weights:
                    self.qnet_hat = self.qnet
                if done:
                    break
