import gymnasium as gym

from gymnasium.envs.registration import register
from enviroment import AllPassFilterEnv

from pathlib import Path
import librosa
from utilities import auto_polarity_detection


# Registering custom enviroment
register(
    # Unique identifier for the env `name-version`
    id="AllPassFilterEnv-v0.1",
    # Path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point='enviroment:AllPassFilterEnv',
    # Max number of steps per episode, using a `TimeLimitWrapper`
    max_episode_steps=100_000,
    # Keyword args for constructor
    kwargs={'input_sig': None, 'target_sig': None, 'fs': None, 'render_mode': 'text', 'seed': None, 'device': None}
)

####### initialize environment hyperparameters ######
env_name = "AllPassFilterEnv-v0.1"

random_seed = 0         # set random seed if required (0 = no random seed)
#####################################################

print("training environment name : " + env_name)

INPUT, FS = librosa.load(Path(f"soundfiles/KickStemIn.wav"), mono=True, sr=None)
TARGET, FS = librosa.load(Path(f"soundfiles/KickStemOut.wav"), mono=True, sr=None)

# Check the polarity of the audio files
POL_INVERT = auto_polarity_detection(INPUT, TARGET)
print("The polarity of the input signal",
    "needs" if POL_INVERT else "does not need",
    "to be inverted.")

env = gym.make(id=env_name,
                input_sig=-INPUT if POL_INVERT else INPUT,
                target_sig=TARGET,
                fs=FS,
                render_mode='text',
                seed=random_seed)