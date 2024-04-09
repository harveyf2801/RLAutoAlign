# Use either a (on/off) Policy Gradient-Based DRL like TRPO, GA3C or PPO
# or a Value-Based DRL like Ape-X DQfD or DARQN to train the network

# DQN / DDQN is usually used for audio based DRL tasks as they're easier
# to implement however, the above algorithms are generally more efficient.

# https://github.com/TianhongDai/reinforcement-learning-algorithms provides
# a variety of DRL algorithms to choose from.


import gymnasium as gym

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env

from enviroment import AllPassFilterEnv

# Parallel environments
# vec_env = make_vec_env("AllPassFilterEnv-v0", n_envs=4)

from pathlib import Path
import librosa
from utilities import auto_polarity_detection

INPUT, FS = librosa.load(Path(f"soundfiles/KickStemIn.wav"), mono=True, sr=None)
TARGET, FS = librosa.load(Path(f"soundfiles/KickStemOut.wav"), mono=True, sr=None)

# Check the polarity of the audio files
POL_INVERT = auto_polarity_detection(INPUT, TARGET)
print("The polarity of the input signal",
    "needs" if POL_INVERT else "does not need",
    "to be inverted.")

vec_env = AllPassFilterEnv(-INPUT if POL_INVERT else INPUT, TARGET, FS, render_mode='text')

# Parallel environments
model = A2C("MultiInputPolicy", vec_env, verbose=1)
model.learn(total_timesteps=25000)
model.save("a2c_apf")

del model # remove to demonstrate saving and loading

model = A2C.load("a2c_apf")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("text")


# model = PPO("MultiInputPolicy", vec_env, verbose=1)
# model.learn(total_timesteps=10)
# model.save("ppo_apf")

# print("*"*8, "DONE", "*"*8)

# del model # remove to demonstrate saving and loading

# model = PPO.load("ppo_apf")

# obs = vec_env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = vec_env.step(action)
#     vec_env.render("text")