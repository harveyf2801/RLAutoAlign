# Use either a (on/off) Policy Gradient-Based DRL like TRPO, GA3C or PPO
# or a Value-Based DRL like Ape-X DQfD or DARQN to train the network

# DQN / DDQN is usually used for audio based DRL tasks as they're easier
# to implement however, the above algorithms are generally more efficient.

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baseline3.common.vec_env import SubprocessVecEnv
from stable_baselines3.common.env_util import make_vec_env

from register_env import register_custom_env

from pathlib import Path
import librosa
from utilities import auto_polarity_detection

import os

# Creating the dirctories to save / load the model
# and driectores to store the log files
models_dir = "models/PPO/"
log_dir = "tmp_logs/"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Creating the enviroment by loading the audio input and target
# then checking the polarity before passing into the enviroment
# registering the enviroment so that it can be used by `gym.make()`
INPUT, FS = librosa.load(Path(f"soundfiles/KickStemIn.wav"), mono=True, sr=None)
TARGET, FS = librosa.load(Path(f"soundfiles/KickStemOut.wav"), mono=True, sr=None)

POL_INVERT = auto_polarity_detection(INPUT, TARGET)
print("The polarity of the input signal",
    "needs" if POL_INVERT else "does not need",
    "to be inverted.")

random_seed = 0 # random seed for reproducibility
env_name = "AllPassFilterEnv-v0.1"

register_custom_env()

vec_env = gym.make(id=env_name,
                input_sig=-INPUT if POL_INVERT else INPUT,
                target_sig=TARGET,
                fs=FS,
                render_mode='text',
                seed=random_seed)

# Creating the Proximal Policy Optimization network and training th model
# 
model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="./board/")

# Training the model, creating a custom callback to save the best model
TIMESTEPS = 10_000

callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
model.learn(total_timesteps=TIMESTEPS)
model.save(f"{models_dir}/{TIMESTEPS}")

print("*"*8, "DONE", "*"*8)

# Deleting the model from memory and loading
# in the model that we've created from storage
del model

model = PPO.load("ppo_apf")

obs = vec_env.reset()
for i in range(3):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("text")



# https://github.com/ClarityCoders/MarioPPO/blob/master/Train.py
# https://www.youtube.com/watch?v=PxoG0A2QoFs