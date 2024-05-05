from typing import SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import Env
import tensorflow as tf

eps = 0.1
gamma = 0.9

env: Env = gym.make("Blackjack-v1")
env.reset()

states = []
pi = {}
q = {}
returns = {}

for i in range(32):
    for j in range(11):
        for k in range(2):
            s = (i, j, k)
            pi[s] = np.random.random()
            q[s] = [0, 0]  # 2 actions
            for a in range(2):
                q[s][a] = 0
                returns[s, a] = []


class QNetwork(tf.keras.Model):
    def __init__(self, num_actions):
        super(QNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(num_actions)

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        return x

# Initialize Q Network
num_actions = 2  # Number of actions in Blackjack
q_network = QNetwork(num_actions)

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

@tf.function
def train_step(states, targets):
    with tf.GradientTape() as tape:
        q_values = q_network(states)
        loss = tf.reduce_mean(tf.square(targets - q_values))
    gradients = tape.gradient(loss, q_network.trainable_variables)
    optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))
    return loss

def generateEpisode():
    env.reset()
    terminated = False
    state, info = env.reset()
    episode = []
    reward = 0
    while not terminated:
        action = np.argmax(q[state]) if np.random.random() > eps else np.argmin(q[state])
        nextstate, reward, terminated, truncated, info = env.step(action)
        episode.append((state, action, reward))
        state = nextstate
    return episode


def learnToPlay():
    episode = generateEpisode()
    g = 0
    for step in range(len(episode) - 1, -1, -1):
        state, action, reward = episode[step]
        g = gamma * g + reward
        returns[state,action].append(g)
        q[state][action] = np.average(returns[state,action])
        pi[state] = np.argmax(q[state])
        targets = q_network(np.array([state])).numpy()
        targets[0][action] = g
        train_step(np.array([state]), np.array(targets))


for i in range(50000):
    learnToPlay()
#
# for k in q.keys():
#     print(k,q[k])
#
#

q_network.save('blackjack_q_network')

for i in range(32):
    for j in range(11):
        for k in range(2):
            state = np.array([[i, j, k]])
            q_values = q_network(state).numpy()
            print(state[0], q_values[0])

