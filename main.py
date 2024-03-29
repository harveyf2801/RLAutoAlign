import numpy as np
import gym
from gym import spaces
from scipy.signal import freqz
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

from Filters.AllPassBand import AllPassBand
from Filters.FilterChain import FilterChain
from Filters.FilterVisualisation import FilterVisualisation


class AllPassFilterEnv(gym.Env):
    def __init__(self, input_sig, target_sig, fs):
        super(AllPassFilterEnv, self).__init__()

        self.fs = fs
        self.input_sig = input_sig
        self.target_sig = target_sig

        self.audio_len = len(self.target_sig)
        self.last_mag = fft((self.target_sig*0.5)+(self.input_sig*0.5), n=self.audio_len)
        self.reward = 0

        band_frequencies = [100, 300, 500, 700, 900]  # Center frequencies of the bands (Hz)
        self.filters = FilterChain([AllPassBand(freq, 0.1, self.fs) for freq in band_frequencies])

        self.visualisation = FilterVisualisation("Filter Response", show_phase=True, show_mag=False)

        # Define action and observation spaces
        self.action_space = spaces.Box(low=np.array([20, 0.1], dtype=np.float32),
                                       high=np.array([800, 10], dtype=np.float32), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([0, 0], dtype=np.float32),
                                            high=np.array([1000, 10], dtype=np.float32), dtype=np.float32)
                
                # Observation space could be phase difference analysis?

    def step(self, action):
        # Update frequency and q values based on the action
        for i, filter_ in enumerate(self.filters):
            filter_.frequency, filter_.q = action[i]
        
        filtered_sig = self.filters.process(self.input_sig)

        # Calculate phase and magnitude differences
        
        # Get an FFT of both signals
        FFT_target = fft(self.target_sig, n=self.audio_len)
        FFT_filtered = fft(filtered_sig, n=self.audio_len)
        # FFT_freqs = fftfreq(self.audio_len, 1 / self.fs)

        # Compute the phase difference
        phase_diff = np.unwrap(np.angle(FFT_filtered)) - np.unwrap(np.angle(FFT_target))

        # Compute the magnitude difference
        current_mag = fft((self.target_sig*0.5)+(filtered_sig*0.5), n=self.audio_len)

        # Update the reward based on the magnitude difference between last and current filter
        mag_diff = current_mag - self.last_mag
        reward = sum(mag_diff)

        # Update the current magnitude
        self.last_mag = current_mag

        # Return a placeholder observation and reward
        obs = np.vstack((phase_diff, current_mag)).T
        done = False
        info = {}

        return obs, reward, done, info

    def reset(self):
        # Reset the environment to its initial state
        band_frequencies = [100, 300, 500, 700, 900]  # Center frequencies of the bands (Hz)
        self.filters = FilterChain([AllPassBand(freq, 0.1, self.fs) for freq in band_frequencies])

        return np.array([[filter_.frequency, filter_.q] for filter_ in self.filters])

    def render(self, mode='text',  **kwargs):

        if mode == 'graph':
            self.visualisation.render(self.filters)

        if mode == 'text':
            for index, filter_ in enumerate(self.filters):
                print("_"*8)
                print(f"Filter {index}: frequency={filter_.frequency}, q={filter_.q}")
                print(f"Magnitude Difference: {self.reward}")
                print("_"*8)

    def close(self):
        pass

# Test the environment
if __name__ == "__main__":
    import librosa
    from pathlib import Path

    INPUT, FS = librosa.load(Path(f"soundfiles/KickStemIn.wav"), mono=True, sr=None)
    TARGET, FS = librosa.load(Path(f"soundfiles/KickStemOut.wav"), mono=True, sr=None)

    mag_1 = 20*np.log10(abs(fft((TARGET*0.5)+(INPUT*0.5), n=len(TARGET))))
    mag_2 = 20*np.log10(abs(fft((TARGET*0.5)+((INPUT*-1)*0.5), n=len(TARGET))))


    # env = AllPassFilterEnv(INPUT, TARGET, FS)
    # obs = env.reset()
    # print("Initial Observation:")
    # print(obs)
    # env.render()

    # # Test with some actions
    # actions = [
    #     [[200, 10], [10000, 10], [20000, 10], [15000, 10], [16000, 10]],
    #     [[200, 10], [300, 10], [400, 10], [500, 10], [600, 10]],
    # ]

    # for action in actions:
    #     obs, _, _, _ = env.step(action)
    #     print("Observation after action:")
    #     print(obs)
    #     env.render()
