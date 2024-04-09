# Use either a (on/off) Policy Gradient-Based DRL like TRPO, GA3C or PPO
# or a Value-Based DRL like Ape-X DQfD or DARQN to train the network

# DQN / DDQN is usually used for audio based DRL tasks as they're easier
# to implement however, the above algorithms are generally more efficient.

# https://github.com/TianhongDai/reinforcement-learning-algorithms provides
# a variety of DRL algorithms to choose from.

import numpy as np

phase_diff = np.array([1, 2, 3])
mag_sum = np.array([6, 7, 8])

result = np.column_stack((phase_diff, mag_sum))

print(result)