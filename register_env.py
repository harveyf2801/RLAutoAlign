from gymnasium.envs.registration import register
from enviroment import AllPassFilterEnv
import gymnasium as gym


# Example for the CartPole environment
def register_custom_env():
    register(
        # unique identifier for the env `name-version`
        id="AllPassFilterEnv-v0.1",
        # path to the class for creating the env
        # Note: entry_point also accept a class as input (and not only a string)
        entry_point='enviroment:AllPassFilterEnv',
        # Max number of steps per episode, using a `TimeLimitWrapper`
        max_episode_steps=10,
        kwargs={'input_sig': None, 'target_sig': None, 'fs': None, 'render_mode': 'text'}
    )

if __name__ == '__main__':
    from pathlib import Path
    import librosa
    from utilities import auto_polarity_detection

    env_name = "AllPassFilterEnv-v0.1"
    random_seed = 0

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
                    render_mode='text')