import matplotlib.pyplot
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
from scipy.signal import freqz
import librosa
import warnings


style.use('dissertation_plot_style.mplstyle')


class Visualisation:
    """Visualisation using matplotlib made to render OpenAI gym environments"""

    def __init__(self, title=None, mode='graph_filters', fs=44100, fps=0.5):
        self.mode = mode
        self.fs = fs
        self.fps = fps

        if self.mode == 'graph_filters':
            self.show_mag = False
            self.show_phase = True

            # Create a figure on and subplot axis and setting a title
            self.fig, self.phase_axis = plt.subplots()

            # Create a twin axes sharing the x-axis
            self.mag_axis = self.phase_axis.twinx()       

        if self.mode == 'observation':
            self.fig, (self.phase_diff_axis, self.db_sum_axis) = plt.subplots(2, 1)
            self.phase_diff_axis.set_title("Unwrapped phase difference")
            self.phase_diff_axis.set_title("Log STFT power")
            self.fig.set_size_inches(15, 10, forward=True)
            self.cb = None

        self.fig.suptitle(title)
        plt.show(block=False)
        warnings.filterwarnings("ignore")

    def _render_phase_response(self, w, h):
        self.phase_axis.clear()
        self.phase_axis.plot(w, np.angle(h, deg=True), label='Phase', color='b')
        self.mag_axis.yaxis.set_label_position('left')
        self.phase_axis.set_ylabel('Phase (deg)')
        self.phase_axis.legend(loc='upper left')
        self.phase_axis.set_ylim([-360, 360])
        self.phase_axis.set_xlim((20, 20000))

    def _render_magnitude_response(self, w, h):
        self.mag_axis.clear()
        self.mag_axis.plot(w, 20*np.log10(np.abs(h)), label='Magnitude', color='r')
        self.mag_axis.yaxis.set_label_position('right')
        self.mag_axis.set_ylabel('Amplitude (dB)')
        self.mag_axis.legend(loc='upper right')
        self.mag_axis.set_ylim((-30, 5))
        self.mag_axis.set_xlim((20, 20000))
    
    def _render_graph_filters(self, filters):
        w, h = filters.get_response()

        if self.show_phase:
            self._render_phase_response(w, h)
        if self.show_mag:
            self._render_magnitude_response(w, h)

        # Setting x scale, limits and ticks
        plt.xlabel('Frequency (Hz)')
        plt.xlim([20, 20000])
        plt.xscale('log')
        plt.xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000],
                ["20", "50", "100", "200", "500", "1k", "2k", "5k", "10k", "20k"])
        
        plt.grid(True)

    def _render_observation(self, obs):
        self.phase_diff_axis.clear()
        self.db_sum_axis.clear()
        
        phaseplot = librosa.display.specshow(obs['phase_diff'], y_axis='log', x_axis='time', ax=self.phase_diff_axis, sr=self.fs)
        dbplot = librosa.display.specshow(obs['db_sum'], y_axis='log', ax=self.db_sum_axis, sr=self.fs)

        if self.cb == None:
            self.cb = self.fig.colorbar(phaseplot, ax=self.phase_diff_axis)
            self.cb = self.fig.colorbar(dbplot, ax=self.db_sum_axis)
    
    def render(self, data):
        if self.mode == 'graph_filters':
            self._render_graph_filters(data)
        
        if self.mode == 'observation':
            self._render_observation(data)

        plt.tight_layout()

        # Necessary to view frames before they are unrendered
        plt.pause(self.fps)

    def close(self):
        plt.close()


if __name__ == '__main__':

    from pathlib import Path
    from utilities import get_magnitude_and_phase_stft

    # Load input and target audio files
    INPUT, FS = librosa.load(Path(f"soundfiles/KickStemIn.wav"), mono=True, sr=None)
    TARGET, FS = librosa.load(Path(f"soundfiles/KickStemOut.wav"), mono=True, sr=None)

    # Compute the spectrogram for both audio files and
    # separate magnitude and phase for both audio files

    S_INPUT, phi_INPUT = get_magnitude_and_phase_stft(INPUT)
    S_TARGET, phi_TARGET = get_magnitude_and_phase_stft(TARGET)

    # Compute the phase difference between the two audio files
    mag_sum = S_INPUT + S_TARGET
    phi_diff = phi_INPUT - phi_TARGET

    Visualisation(title = "Enviroment",
                        mode='observation').render({'phase_diff': phi_diff, 'db_sum': librosa.core.amplitude_to_db((mag_sum)**2, ref=np.max)})