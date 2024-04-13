# Use either a (on/off) Policy Gradient-Based DRL like TRPO, GA3C or PPO
# or a Value-Based DRL like Ape-X DQfD or DARQN to train the network

# DQN / DDQN is usually used for audio based DRL tasks as they're easier
# to implement however, the above algorithms are generally more efficient.

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

from stablebaseline_callbacks import SaveOnBestTrainingRewardCallback

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
        # Creating the enviroment by loading the audio input and target
        # then checking the polarity before passing into the enviroment
        # registering the enviroment so that it can be used by `gym.make()`
        INPUT, FS = librosa.load(Path("soundfiles/KickStemIn.wav"), mono=True, sr=None)
        TARGET, FS = librosa.load(Path("soundfiles/KickStemOut.wav"), mono=True, sr=None)

        POL_INVERT = auto_polarity_detection(INPUT, TARGET)
        print("The polarity of the input signal",
            "needs" if POL_INVERT else "does not need",
            "to be inverted.")
        
        env = gym.make(
                    id=env_id,
                    input_sig=-INPUT if POL_INVERT else INPUT,
                    target_sig=TARGET,
                    fs=FS,
                    render_mode='text')
        # env = TimeLimitWrapper(env, max_steps=2000)
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    # Creating the dirctories to save / load the model
    # and driectores to store the log files
    models_dir = Path("models", "PPO")
    log_dir = Path("tmp")

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    seed = 0 # random seed for reproducibility
    env_name = "AllPassFilterEnv-v0.1"

    # Creating multiple enviroments
    # and wrapping it in a multiprocessed vectorised wrapper,
    # then wrapping this in a monitor
    vec_env = VecMonitor(SubprocVecEnv([make_env(env_name, i, seed) for i in range(1)]),
                    filename="tmp/TestMonitor")

    # Creating the Proximal Policy Optimization network
    model = PPO("MlpPolicy", vec_env, n_steps=10, seed=seed, verbose=1, tensorboard_log="./board/")

    # Training the model, creating a custom callback to save the best model
    print("*"*8, "Training", "*"*8)
    TIMESTEPS = 100
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    model.learn(total_timesteps=TIMESTEPS, callback=callback, tb_log_name="PPO_APF", progress_bar=True)
    model.save(Path(models_dir, env_name))

    print("*"*8, "DONE", "*"*8)

    # Deleting the model from memory and loading
    # in the model that we've created from storage
    # del model

    # model = PPO.load("ppo_apf")

    # obs = vec_env.reset()
    # for i in range(3):
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = vec_env.step(action)
    #     vec_env.render("text")



# https://github.com/ClarityCoders/MarioPPO/blob/master/Train.py
# https://www.youtube.com/watch?v=PxoG0A2QoFs

# tensorboard --logdir ./board/ --host localhost --port 8088