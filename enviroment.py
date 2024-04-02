import numpy as np
import gym
from gym import spaces
from scipy.signal import freqz
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import librosa


from Filters.AllPassBand import AllPassBand
from Filters.FilterChain import FilterChain
from Filters.FilterVisualisation import FilterVisualisation


class AllPassFilterEnv(gym.Env):
    def __init__(self, input_sig, target_sig, fs, plotting_visuals=False):
        super(AllPassFilterEnv, self).__init__()

        self.fs = fs
        self.input_sig = input_sig
        self.target_sig = target_sig

        self.audio_len = len(self.target_sig)
        self.original_rms = np.mean(librosa.feature.rms(y=(0.5*self.input_sig)+(0.5*self.target_sig)))
        self.reward = 0

        band_frequencies = [100, 300, 500, 700, 900]  # Center frequencies of the bands (Hz)
        self.filters = FilterChain([AllPassBand(freq, 0.1, self.fs) for freq in band_frequencies])

        if (plotting_visuals):
            self.visualisation = FilterVisualisation("Filter Response", show_phase=True, show_mag=False)

        # Define action and observation spaces
        self.action_space = spaces.Box(low=np.array([20, 0.1], dtype=np.float32),
                                       high=np.array([800, 10], dtype=np.float32), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([-360, -100], dtype=np.float32),
                                            high=np.array([360, 100], dtype=np.float32), dtype=np.float32)
                
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

        # Compute the phase difference for observation space
        phase_diff = np.angle(FFT_filtered, deg=True) - np.angle(FFT_target, deg=True)

        summed_sig = (0.5*self.target_sig)+(0.5*filtered_sig)

        # Compute dB  single sided FFT values for the observation space
        FFT_current = fft(summed_sig, n=self.audio_len)
        FFT_current = 20*np.log10(2.0/self.audio_len * np.abs(FFT_current))#[:self.audio_len//2]))

        # Compute the rms difference for the reward
        rms = np.mean(librosa.feature.rms(y=summed_sig))
        self.reward = rms - self.original_rms

        # Return a placeholder observation and reward
        # states = np.array([[filter_.frequency, filter_.q] for filter_ in self.filters])
        obs = np.vstack((phase_diff, FFT_current)).T
        done = False
        info = {"RMS": rms, "Reward": self.reward}

        self.render()

        # state/obs, reward, terminated/done, 
        return obs, self.reward, done, info

    def reset(self):
        # Reset the environment to its initial state
        band_frequencies = [100, 300, 500, 700, 900]  # Center frequencies of the bands (Hz)
        self.filters = FilterChain([AllPassBand(freq, 0.1, self.fs) for freq in band_frequencies])

        self.render()

        return np.array()

    def render(self, mode='text',  **kwargs):

        if mode == 'graph':
            self.visualisation.render(self.filters)

        if mode == 'text':
            print("_"*8)
            for index, filter_ in enumerate(self.filters):
                print(f"Filter {index}: frequency={filter_.frequency}, q={filter_.q}")
            print(f"RMS Difference: {self.reward}")
            print("_"*8)

    def close(self):
        pass

# Test the environment
if __name__ == "__main__":

    from pathlib import Path

    INPUT, FS = librosa.load(Path(f"soundfiles/KickStemIn.wav"), mono=True, sr=None)
    TARGET, FS = librosa.load(Path(f"soundfiles/KickStemOut.wav"), mono=True, sr=None)

    env = AllPassFilterEnv(INPUT, TARGET, FS)
    obs = env.reset()
    print("Initial Observation:")
    print(obs)
    env.render()

    # Test with some actions
    actions = [
        [[200, 10], [10000, 10], [20000, 10], [15000, 10], [16000, 10]],
        [[100, 0.5], [500, 1], [1600, 10], [2300, 2], [3000, 6]],
    ]

    for action in actions:
        obs, _, _, _ = env.step(action)
        print("Observation after action:")
        print(np.max(obs))
        env.render()
