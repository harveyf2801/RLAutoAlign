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
    """
    ### Description

    The AllPassFilterEnv is a gymnasium enviroment that consists of 5 2nd order IIR all-pass filters.
    These filters are applied using Transposed Direct Form II (TDF2) and each have frequency and Q
    parameters that can be altered via the continuous action space. The goal of enviroment is to find the optimal
    parameters to align an input signal with a target signal audio. The reward is based on the RMS difference between
    the original input + target signal, and the filtered signal + target signal.

    ### Observation Space

    The observation is a flattened `ndarray` where the first half of the array is the phase difference between the
    filtered signal and the target signal, and the second half is the sum of the magnitudes of the filtered signal and
    the target signal.

    ``

    ### Action Space

    The action space consists of a `ndarray` with the flattened shape from (5, 2) to (1, ) with continuous actions
    that represent the frequency and Q values of the 5 all-pass filters. These parameters are normalised but then
    remapped to the ranges of [20, 20kHz] for frequency and the [0.1, 10] for the Q.

    ### Reward:

    The reward is based on the RMS difference between the original input + target signal,
    and the filtered signal + target signal. This was chosen as when aligning the audio signals, the bi-product
    of the constructive interference is an overall louder signal. 

    ### Starting State

    The parameters are all randomly selected when the enviroment is initially reset, using the seed for reproducibility.
    The observation is then calculated for the initial state.

    ### Episode End

    The episode ends if either of the following happens:
    1. Termination: The reward/RMS difference hits +20 (as this is a good enough alignment for the audio signals)
    2. Truncation: The length of the episode is 10_000 steps.


    ### Arguments

    ```
    gym.make('AllPassFilterEnv-v0.1')
    ```

    ### Version History

    * v0.1: Initial version
    """

    metadata = {"render_modes": ["text", "graph_filters", "observation"], "render_fps": 0.5}

    def __init__(self, input_sig, target_sig, fs, render_mode='text'):
        super(AllPassFilterEnv, self).__init__()
        self.steps = 0

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

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        if (self.render_mode != 'text'):
            self.visualisation = Visualisation("DRL Visuals", self.render_mode, fs=self.fs, fps=self.metadata["render_fps"])

        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_filterbands * 2,), dtype=np.float32)
        
        obs_shape = self._get_observation_size()

        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(2 * obs_shape[0]*obs_shape[1],),
                                            dtype=np.float64)

        # These get generated on the `reset()` call
        self.filters = None
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
        action = action.reshape(5, 2)

        # Update frequency and q values based on the action
        for i, filter_ in enumerate(self.filters):

            # Map action to freq and q ranges
            freq = np.interp(action[i][0], [-1, 1], [self.frequency_range[0], self.frequency_range[1]])
            q = np.interp(action[i][1], [-1, 1], [self.q_range[0], self.q_range[1]])

            # Update filter params
            filter_.frequency, filter_.q = freq, q
    
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
        return np.concatenate((phase_diff.ravel(), mag_sum.ravel()))

    def _get_info(self):
        info = {}
        for i, filter_ in enumerate(self.filters):
            info[f"Filter {i}"] = {'Frequency': filter_.frequency, 'Q': filter_.q}
        return info

    def step(self, action):
        self.steps += 1

        # Updating the filter from the steps actiona and filtering the input signal
        self._update_filter_chain(action)
        filtered_sig = self._get_filtered_signal()

        # Compute the rms difference for the reward
        rms = get_rms_decibels(filtered_sig+self.target_sig)
        self.reward = rms - self.original_rms
        terminated = bool(self.reward > 20) # if the reward is over 20dB RMS, the episode is done
        # truncated = bool(self.steps > 10_000) # if the steps reach 10,000 max steps

        # self.render()

        # observation, reward, terminated, False, info
        return self._get_obs(filtered_sig), self.reward, terminated, False, self._get_info()

    def reset(self, seed=None, options=None):
        # To seed self.np_random
        super().reset(seed=seed)
                  
        # Reset the environment to its initial state
        self.filters = FilterChain([AllPassBand(self.np_random.uniform(*self.frequency_range), self.np_random.uniform(*self.q_range), self.fs) for _ in range(self.n_filterbands)])

        filtered_sig = self._get_filtered_signal()
        observation = self._get_obs(filtered_sig)
        info = self._get_info()

        # self.render()

        return observation, info

    def render(self, **kwargs):

        if self.render_mode == 'text':
            print("_"*8)
            for index, filter_ in enumerate(self.filters):
                print(f"Filter {index}: frequency={filter_.frequency}, q={filter_.q}")
            print(f"RMS Difference: {self.reward}")
            print(f"Step: {self.steps}")
            print("_"*8)
        
        elif self.render_mode == 'graph_filters':
            self.visualisation.render(self.filters)

        elif self.render_mode == 'observation':
            self.visualisation.render(self.current_obs)

    def close(self):
        if self.render_mode != 'text':
            self.visualisation.close()

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

    env = AllPassFilterEnv(-INPUT if POL_INVERT else INPUT, TARGET, FS, render_mode='text')

    obs, info = env.reset()

    sample = env.action_space.sample()
    _obs, reward, terminated, truncated, info = env.step(sample)

    print(obs.shape, env.observation_space.shape)

    # check_gym_env(env)

    # obs, info = env.reset()

    # # Test with some actions
    # actions = [
    #     [[100, 10], [102, 10], [600, 10], [605, 10], [2500, 10]],
    #     [[100, 0.5], [500, 1], [1600, 10], [2300, 2], [3000, 6]],
    # ]

    # for action in actions:
    #     _, _, _, _, info = env.step(action)