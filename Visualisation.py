import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
from scipy.signal import freqz

style.use('dissertation_plot_style.mplstyle')


class Visualisation:
    """Visualisation using matplotlib made to render OpenAI gym environments"""

    def __init__(self, title=None, show_mag=True, show_phase=False):

        self.show_mag = show_mag
        self.show_phase = show_phase

        # Create a figure on and subplot axis and setting a title
        self.fig, self.phase_axis = plt.subplots()
        self.fig.suptitle(title)

        # Create a twin axes sharing the x-axis
        self.mag_axis = self.phase_axis.twinx()       

        plt.show(block=False)

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

    def render(self, filters):
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
        plt.tight_layout()

        # Necessary to view frames before they are unrendered
        plt.pause(5)

    def close(self):
        plt.close()


if __name__ == '__main__':
    from Filters.AllPassBand import AllPassBand
    from Filters.FilterChain import FilterChain

    FS = 44100
    chain = FilterChain([AllPassBand(200, 0.5, FS), AllPassBand(10000, 2, FS)])
    Visualisation(title = "Filter Response",
                        show_phase=True,
                        show_mag=False).render(chain)