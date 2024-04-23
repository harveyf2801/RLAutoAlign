from gymnasium.envs.registration import register
from enviroment import AllPassFilterEnv
import gymnasium as gym


# Example for the CartPole environment
def register_custom_env():
    register(
        # unique identifier for the env `name-version`
        id="AllPassFilterEnv-v0.2",
        # path to the class for creating the env
        # Note: entry_point also accept a class as input (and not only a string)
        entry_point='enviroment:AllPassFilterEnv',
        kwargs={'audio_dir': None, 'render_mode': 'text'}
    )
