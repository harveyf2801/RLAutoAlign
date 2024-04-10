import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.signal import freqz, csd
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import librosa

from utilities import get_rms_decibels, get_magnitude_and_phase_stft

from Filters.AllPassBand import AllPassBand
from Filters.FilterChain import FilterChain
from Visualisation import Visualisation


class AllPassFilterEnv(gym.Env):
    metadata = {"render_modes": ["text", "graph_filters", "observation"], "render_fps": 4}

    def __init__(self, input_sig, target_sig, fs, render_mode='text', seed=1, device=None):
        super(AllPassFilterEnv, self).__init__()

        self.device = device

        self.seed = seed # Set the seed for reproducibility

        self.fft_size=1024
        self.hop_size=256
        self.win_length=1024

        self.fs = fs
        self.input_sig = input_sig#[:fs*10]
        self.target_sig = target_sig#[:fs*10]
        
        self.original_rms = get_rms_decibels(self.input_sig+self.target_sig)
        self.reward_range = (-80, 80)
        self.reward = 0
        
        self.n_filterbands = 5
        self.frequency_range = (1, 800)
        self.q_range = (0.1, 10)

        self.filters = FilterChain([AllPassBand(self.np_random.uniform(*self.frequency_range), self.np_random.uniform(*self.q_range), self.fs) for _ in range(self.n_filterbands)])

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        if (self.render_mode != 'text'):
            self.visualisation = Visualisation("DRL Visuals", self.render_mode, fs=self.fs, fps=self.metadata["render_fps"])

        # Define action and observation spaces
        self.action_space = spaces.Box(low=np.array([(20, 0.1)]*5, dtype=np.float32),
                               high=np.array([(800, 10)]*5, dtype=np.float32), shape=(5, 2), dtype=np.float32)
        
        obs_shape = self._get_observation_size()

        self.observation_space = spaces.Dict({
        'phase_diff': spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float64),
        'db_sum': spaces.Box(low=-80, high=0, shape=obs_shape, dtype=np.float64)
        })

        self.current_obs = None
    
    def _get_observation_size(self):
        signal_length = len(self.input_sig)

        # Calculate the number of windows
        num_windows = 1 + (signal_length - self.win_length) // self.hop_size
        if (signal_length - self.win_length) % self.hop_size != 0:
            num_windows += 1

        # Add 3 to account for padding at the beginning and end of the signal
        num_windows += 3

        # The number of frequency bins is half the window size plus one
        num_freq_bins = self.win_length // 2 + 1

        # The shape of the magnitude and phase arrays is (num_windows, num_freq_bins)
        return (num_freq_bins, num_windows)

    def _update_filter_chain(self, action):
        # Update frequency and q values based on the action
        for i, filter_ in enumerate(self.filters):
            filter_.frequency, filter_.q = action[i][0], action[i][1]
    
    def _get_filtered_signal(self):
        return self.filters.process(self.input_sig)
    
    def _get_obs(self, filtered_sig):
        # Get the magnitude and phase (rad) of both signals
        T_mag, T_phase = get_magnitude_and_phase_stft(self.target_sig,
            self.fft_size,
            self.hop_size,
            self.win_length)
        X_mag, X_phase = get_magnitude_and_phase_stft(filtered_sig,
            self.fft_size,
            self.hop_size,
            self.win_length)

        # Compute the phase difference and magnitude sum for observation space
        phase_diff = X_phase - T_phase
        mag_sum = X_mag + T_mag

        # Compute dB single sided FFT values for the observation space
        db_sum = librosa.core.amplitude_to_db((mag_sum)**2, ref=np.max)

        # Return a placeholder observation and reward
        self.current_obs = {'phase_diff': phase_diff, 'db_sum': db_sum}
        return self.current_obs

    def _get_info(self):
        info = {}
        for i, filter_ in enumerate(self.filters):
            info[f"Filter {i}"] = {'Frequency': filter_.frequency, 'Q': filter_.q}
        return info

    def step(self, action):
        # Updating the filter from the steps actiona and filtering the input signal
        self._update_filter_chain(action)
        filtered_sig = self._get_filtered_signal()

        # Compute the rms difference for the reward
        rms = get_rms_decibels(filtered_sig+self.target_sig)
        self.reward = rms - self.original_rms
        terminated = self.reward > 0 # if the reward is over 4dB RMS, the episode is done

        self.render()

        # observation, reward, terminated, False, info
        return self._get_obs(filtered_sig), self.reward, bool(terminated), False, self._get_info()

    def reset(self, seed=None, options=None):
        # To seed self.np_random
        super().reset(seed=seed)
                  
        # Reset the environment to its initial state
        self.filters = FilterChain([AllPassBand(self.np_random.uniform(*self.frequency_range), self.np_random.uniform(*self.q_range), self.fs) for _ in range(self.n_filterbands)])

        filtered_sig = self._get_filtered_signal()
        observation = self._get_obs(filtered_sig)
        info = self._get_info()

        self.render()

        return observation, info

    def render(self, **kwargs):

        if self.render_mode == 'text':
            print("_"*8)
            for index, filter_ in enumerate(self.filters):
                print(f"Filter {index}: frequency={filter_.frequency}, q={filter_.q}")
            print(f"RMS Difference: {self.reward}")
            print("_"*8)
        
        elif self.render_mode == 'graph_filters':
            self.visualisation.render(self.filters)

        elif self.render_mode == 'observation':
            self.visualisation.render(self.current_obs)

    def close(self):
        if self.render_mode != 'text':
            self.visualisation.close()



def register_env():
    from gymnasium.envs.registration import register
    # Example for the CartPole environment
    register(
        # unique identifier for the env `name-version`
        id="AllPassFilterEnv-v0",
        # path to the class for creating the env
        # Note: entry_point also accept a class as input (and not only a string)
        entry_point="gym.envs.harveyf2801:AllPassFilterEnv",
        # Max number of steps per episode, using a `TimeLimitWrapper`
        max_episode_steps=100_000,
    )

def check_gym_env(env):
    from stable_baselines3.common.env_checker import check_env

    # It will check your custom environment and output additional warnings if needed
    check_env(env)

# Test the environment
if __name__ == "__main__":

    from pathlib import Path
    from utilities import auto_polarity_detection

    INPUT, FS = librosa.load(Path(f"soundfiles/KickStemIn.wav"), mono=True, sr=None)
    TARGET, FS = librosa.load(Path(f"soundfiles/KickStemOut.wav"), mono=True, sr=None)

    # Check the polarity of the audio files
    POL_INVERT = auto_polarity_detection(INPUT, TARGET)
    print("The polarity of the input signal",
        "needs" if POL_INVERT else "does not need",
        "to be inverted.")

    env = AllPassFilterEnv(-INPUT if POL_INVERT else INPUT, TARGET, FS, render_mode='graph_filters')
    
    check_gym_env(env)

    # obs, info = env.reset()

    # # Test with some actions
    # actions = [
    #     [[100, 10], [102, 10], [600, 10], [605, 10], [2500, 10]],
    #     [[100, 0.5], [500, 1], [1600, 10], [2300, 2], [3000, 6]],
    # ]

    # for action in actions:
    #     _, _, _, _, info = env.step(action)