# Use either a (on/off) Policy Gradient-Based DRL like TRPO, GA3C or PPO
# or a Value-Based DRL like Ape-X DQfD or DARQN to train the network

# DQN / DDQN is usually used for audio based DRL tasks as they're easier
# to implement however, the above algorithms are generally more efficient.

import gymnasium as gym
from enviroment import TimeLimitWrapper

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback

from stablebaseline_callbacks import SummaryWriterCallback

from register_env import register_custom_env

from pathlib import Path
import librosa
from utilities import auto_polarity_detection

import os

register_custom_env()

def make_env(env_id, i, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param i: (int) index of the subprocess
    """
    def _init():
        # Creating the enviroment        
        env = gym.make(
                    id=env_id,
                    audio_dir='/home/',
                    render_mode='text')
        env = Monitor(TimeLimitWrapper(env, max_steps=20_000),
                      filename="tmp/TestMonitor")
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    # Creating the dirctories to save / load the model
    # and driectores to store the log files
    models_dir = Path("models", "PPO")
    log_dir = Path("tmp", "logs")

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    seed = 0 # random seed for reproducibility
    env_name = "AllPassFilterEnv-v0.2"

    # Creating multiple enviroments
    # and wrapping it in a multiprocessed vectorised wrapper,
    # then wrapping this in a monitor

    # vec_env = VecMonitor(SubprocVecEnv([make_env(env_name, i, seed) for i in range(1)]),
    #                 filename="tmp/TestMonitor")

    # In this case to save on memory we're only using one environment
    vec_env = make_env(env_name, 0, seed)()

    # Creating the Proximal Policy Optimization network
    model = PPO("MlpPolicy", vec_env, seed=seed, verbose=1, tensorboard_log='./board/')
    
    # Load the pre-trained model
    model_path = Path(models_dir, "ppo_apf_3900000_steps.zip")
    model.load(model_path)

    # Test the model on the environment
    import tqdm
    TOTAL_TIMESTEPS = 10
    obs, info = vec_env.reset()
    with tqdm.tqdm(total=TOTAL_TIMESTEPS) as pbar:
        for i in range(TOTAL_TIMESTEPS):
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, information = vec_env.step(action)
            vec_env.render()
            pbar.update(1)
    vec_env.close()

    #