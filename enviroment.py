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
    def __init__(self, input_sig, target_sig, fs, plotting_visuals=False):
        super(AllPassFilterEnv, self).__init__()

        self.fs = fs
        self.input_sig = input_sig
        self.target_sig = target_sig
        
        self.original_rms = get_rms_decibels(self.input_sig+self.target_sig)
        self.reward = 0
        self.reward_range = (-80, 80)

        band_frequencies = [100, 300, 500, 700, 900]  # Center frequencies of the bands (Hz)
        self.filters = FilterChain([AllPassBand(freq, 0.1, self.fs) for freq in band_frequencies])

        self.plotting_visuals = plotting_visuals
        if (self.plotting_visuals):
            self.visualisation = Visualisation("DRL Visuals", show_phase=True, show_mag=False)

        # Define action and observation spaces
        self.action_space = spaces.Box(low=np.array([20, 0.1]*5, dtype=np.float32),
                               high=np.array([800, 10]*5, dtype=np.float32), dtype=np.float32)
        
        self.observation_space = spaces.Dict({
        'phase_diff': spaces.Box(low=-np.pi, high=np.pi, shape=(len(band_frequencies),), dtype=np.float32),
        'db_sum': spaces.Box(low=-80, high=0, shape=(len(band_frequencies),), dtype=np.float32)
        })

    def step(self, action):
        # Update frequency and q values based on the action
        for i, filter_ in enumerate(self.filters):
            filter_.frequency, filter_.q = action[i][0], action[i][1]
        
        filtered_sig = self.filters.process(self.input_sig)

        # Get the magnitude and phase (rad) of both signals
        T_mag, T_phase = get_magnitude_and_phase_stft(self.target_sig)
        X_mag, X_phase = get_magnitude_and_phase_stft(filtered_sig)

        # Compute the phase difference and magnitude sum for observation space
        phase_diff = X_phase - T_phase
        mag_sum = X_mag + T_mag

        # Compute dB single sided FFT values for the observation space
        db_sum = librosa.core.amplitude_to_db((mag_sum)**2, ref=np.max)

        # Compute the rms difference for the reward
        rms = get_rms_decibels(filtered_sig+self.target_sig)
        self.reward = rms - self.original_rms

        # Return a placeholder observation and reward
        obs = {'phase_diff': phase_diff, 'db_sum': db_sum}
        done = self.reward < 0 # if the reward is negative, the episode is done
        info = {"RMS": rms, "Reward": self.reward}

        self.render()

        # state/obs, reward, terminated/done, 
        return obs, self.reward, done, info

    def reset(self):
        # Reset the environment to its initial state
        band_frequencies = [100, 300, 500, 700, 900]  # Center frequencies of the bands (Hz)
        self.filters = FilterChain([AllPassBand(freq, 0.1, self.fs) for freq in band_frequencies])

        self.render()

        return self.observation_space

    def render(self, mode='text',  **kwargs):

        if mode == 'graph_filters':
            self.visualisation.render(self.filters)

        if mode == 'text':
            print("_"*8)
            for index, filter_ in enumerate(self.filters):
                print(f"Filter {index}: frequency={filter_.frequency}, q={filter_.q}")
            print(f"RMS Difference: {self.reward}")
            print("_"*8)

    def close(self):
        if self.plotting_visuals:
            self.visualisation.close()

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

    env = AllPassFilterEnv(-INPUT if POL_INVERT else INPUT, TARGET, FS)
    obs = env.reset()

    # Test with some actions
    actions = [
        [[200, 10], [10000, 10], [20000, 10], [15000, 10], [16000, 10]],
        [[100, 0.5], [500, 1], [1600, 10], [2300, 2], [3000, 6]],
    ]

    for action in actions:
        print("Observation after action:")
        obs, _, _, _ = env.step(action)