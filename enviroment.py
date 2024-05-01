import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.signal import freqz, csd
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import librosa
from pathlib import Path
import logging
from losses import MultiResolutionSTFTLoss

from utilities import get_rms_decibels, get_magnitude_and_phase_stft, auto_polarity_detection

from annotations import get_annotations

from Filters.AllPassBand import AllPassBand
from Filters.FilterChain import FilterChain
from Visualisation import Visualisation

from losses import MultiResolutionSTFTLoss
from evaluation_metrics import mse_loss, db_rms, dB_peak, thdn


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

    The reward is based on the inverse MR-STFT loss difference between the filtered signal + target signal. This was chosen as
    when aligning the audio signals, the bi-product of the constructive interference is an overall louder signal. Therefore,
    the stft loss will compare the sum of the two magnitudes of the individual signals with the magnitude of the summed signals
    before taking the stft. This loss can go between 510 to 0 where 0 is the perfectly aligned signal and 510 is completely out
    of phase.

    ### Starting State

    The parameters are all randomly selected when the environment is initially reset, using the seed for reproducibility.
    The observation is then calculated for the initial state.

    ### Episode End

    The episode ends if either of the following happens:
    1. Termination: The reward/RMS difference hits +20 (as this is a good enough alignment for the audio signals)
    2. Truncation: The length of the episode is 200 steps.


    ### Arguments

    ```
    gym.make('AllPassFilterEnv-v0.1')
    ```

    ### Version History

    * v0.1: Initial version using RMS difference with 5 filter bands between 20 and 800 Hz.
    * v0.2: Started using the inverse MR-STFT loss as the reward and for polarity detection.
    """

    metadata = {"render_modes": ["text", "graph_filters", "observation"], "render_fps": 0.5}

    def __init__(self, audio_dir, render_mode='text'):
        super(AllPassFilterEnv, self).__init__()

        # Feature extraction params

        self.fft_size=1024
        self.hop_size=256
        self.win_length=1024
        self.window="hann_window",
        self.eps=1e-8,

        self.fs = 44100 # target samplerate
        self.audio_length = 0.5 # target seconds

        self.annotations = get_annotations(audio_dir)
        self.audio_dir = audio_dir
        self.max_class_id = max(self.annotations.ClassID) # number of classes
        
        self.loss = MultiResolutionSTFTLoss()
        # self.reward_range = (-1, 1)
        
        self.n_filterbands = 10
        self.frequency_range = (20, 20000)
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
                                            dtype=np.float32)

        # These get generated on the `reset()` call
        self.filters = None
        self.current_obs = None
        self.reward = None
        self.original_loss = None

        self.total_steps = 0

        # Metrics
        self.losses = {
            'MR-STFT': MultiResolutionSTFTLoss(),
            'MSE': mse_loss
        }

        self.loudness = {
            'RMS': db_rms,
            'Peak': dB_peak
        }

        self.quality = {
            'THDN': thdn
        }

        self.total_loss = {'MR-STFT': 0, 'MSE': 0}
        self.total_loudness = {'RMS': 0, 'Peak': 0}
        self.total_quality = {'THDN': 0}

        self._reset_audio()
    
    def _choose_random_TB_pair(self):
        # Selecting a random class
        class_id = self.np_random.integers(0, self.max_class_id)
        # Selecting a random top and bottom snare record from annotations
        # input_df = self.annotations.query(f'(ClassID == {class_id}) & (Position == "SHL")').sample(n=2)
        # target_df = self.annotations.query(f'(ClassID == {class_id}) & (Position == "SHL")').sample(n=1)
        df = self.annotations.query(f'(ClassID == {class_id}) & (Position == "SHL")').sample(n=2)
        input_df = df.iloc[0]
        target_df = df.iloc[1]

        return input_df, target_df
    
    def _load_audio_files(self, input_filename, target_filename):
        # Load in audio with a target samplerate, duration and channels
        input_sig, fs = librosa.load(Path(self.audio_dir, input_filename), mono=True, sr=self.fs, duration=self.audio_length)
        target_sig, _fs = librosa.load(Path(self.audio_dir, target_filename), mono=True, sr=self.fs, duration=self.audio_length)

        # Normalize the audio signals
        input_sig /= np.max(np.abs(input_sig))
        target_sig /= np.max(np.abs(target_sig))

        # Check the polarity of the audio files
        pol_invert = abs(float(self.loss(input_sig, target_sig))) > abs(float(self.loss(-input_sig, target_sig)))
        
        return -input_sig if pol_invert else input_sig, target_sig
    
    def _get_observation_size(self):
        signal_length = int(self.fs * self.audio_length)

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
        action = action.reshape(self.n_filterbands, 2)

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
        phase_diff = np.array(X_phase - T_phase, dtype=np.float32)
        mag_sum = np.array(X_mag + T_mag, dtype=np.float32)

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

    def _get_reward(self, x, y):
        loss = abs(float(self.loss(x, y)))
        reward = self.original_loss - loss
        return reward

    def step(self, action):

        # Updating the filter from the steps action and filtering the input signal
        self._update_filter_chain(action)
        filtered_sig = self._get_filtered_signal()

        mix = (self.target_sig + filtered_sig) / 2
        for loss_key, loss in self.losses.items():
            self.total_loss[loss_key] += float(loss(self.target_sig, filtered_sig))
        for loudness_key, loudness_func in self.loudness.items():
            self.total_loudness[loudness_key] += loudness_func(mix)
        for quality_key, quality_func in self.quality.items():
            self.total_quality[quality_key] += quality_func(self.target_sig, filtered_sig)

        self.total_steps += 1

        # Compute the rms difference for the reward
        self.reward = 1#self._get_reward(filtered_sig, self.target_sig)
        # terminated = bool((reward > ? or reward < ?)

        # self.render()

        # observation, reward, terminated, False, info
        return self._get_obs(filtered_sig), self.reward, False, False, self._get_info()
    
    def _reset_audio(self):
        # Reset the environment to its initial state
        self.input_df, self.target_df = self._choose_random_TB_pair() # selecting a random top and bottom snare
        self.input_sig, self.target_sig = self._load_audio_files(self.input_df['FileName'],
                                                                self.target_df['FileName']) # load audio files with checks

        self.original_loss = abs(float(self.loss(self.input_sig, self.target_sig)))
        # print('Original Reward: ', self.original_loss)

        # print(f"Audio Reset:\n{self.input_df['FileName']}\n{self.target_df['FileName']}")

    def reset(self, seed=None, options=None):
        # To seed self.np_random
        super().reset(seed=seed)

        self._reset_audio()

        self.filters = FilterChain([AllPassBand(self.np_random.uniform(*self.frequency_range), self.np_random.uniform(*self.q_range), self.fs) for _ in range(self.n_filterbands)])

        filtered_sig = self._get_filtered_signal()
        observation = self._get_obs(filtered_sig)

        self.reward = self._get_reward(filtered_sig, self.target_sig)

        info = self._get_info()
        # self.render()

        return observation, info

    def render(self, **kwargs):

        if self.render_mode == 'text':
            print("_"*8)
            for index, filter_ in enumerate(self.filters):
                print(f"Filter {index}: frequency={filter_.frequency}, q={filter_.q}")
            print(f"Reward: {self.reward}")
            print("_"*8)
        
        elif self.render_mode == 'graph_filters':
            self.visualisation.render(self.filters)

        elif self.render_mode == 'observation':
            self.visualisation.render(self.current_obs)

    def close(self):
        if self.render_mode != 'text':
            self.visualisation.close()

class TimeLimitWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    :param max_steps: (int) Max number of steps per episode
    """

    def __init__(self, env, max_steps=100):
        # Call the parent constructor, so we can access self.env later
        super(TimeLimitWrapper, self).__init__(env)
        self.max_steps = max_steps
        self.current_episode = 0
        # Counter of steps per episode
        self.current_step = 0

    def reset(self, **kwargs):
        """
        Reset the environment
        """
        # Reset the counter
        self.current_step = 0

        # print('Episode: ', self.current_episode)
        self.current_episode += 1
        return self.env.reset(**kwargs)

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, bool, dict) observation, reward, is the episode over?, additional informations
        """
        self.current_step += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Overwrite the truncation signal when when the number of steps reaches the maximum
        if self.current_step >= self.max_steps:
            truncated = True
        
        # print('Step: ', self.current_step)
        return obs, reward, terminated, truncated, info
    
def check_gym_env(env):
    from stable_baselines3.common.env_checker import check_env

    # It will check your custom environment and output additional warnings if needed
    check_env(env)

# Test the environment
if __name__ == "__main__":

    env = AllPassFilterEnv('/home/hf1/Documents/soundfiles/SDDS_segmented_Allfiles')
    
    # env = AllPassFilterEnv('C:/Users/hfret/Downloads/SDDS')
    check_gym_env(env)

    # obs, info = env.reset()

    # # Test with some actions
    # actions = [
    #     np.array([1, 0.1, 1, 0.2, 1, 0.3, 1, 0.4, 1, 0.5]),
    #     np.array([0.6, 0.6, 0.7, 0.7, 0.8, 0.8, 0.9, 0.9, 1, 1]),
    # ]

    # for action in actions:
    #     _, _, _, _, info = env.step(action)