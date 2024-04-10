from scipy.signal import freqz, lfilter
import numpy as np

class FilterChain:
    """
    Filter Chain class that chains multiple FilterBand objects together.
    """

    def __init__(self, filter_bands, fs=None):
        self.filter_bands = filter_bands
        self.fs = fs if fs else filter_bands[0].fs

    def add_filter_band(self, filter_band):
        """
        Add a FilterBand object to the filter chain.
        """
        self.filter_bands.append(filter_band)
    
    def remove_filter_band(self, index):
        del self.filter_bands[index]
    
    def get_response(self):
        b_total = 1
        a_total = 1

        for filter_ in self.filter_bands:
            b_total = np.convolve(b_total, filter_.b)
            a_total = np.convolve(a_total, filter_.a)

        return freqz(b_total, a_total, worN=8000, fs=self.fs)
    
    @property
    def phase_response(self):
        """
        Get the phase response of the filter.
        """
        w, h = self.get_response()
        return w, np.unwrap(np.angle(h, deg=True))
    
    @property
    def magnitude_response(self):
        """
        Get the magnitude response of the filter.
        """
        w, h = self.get_response()
        return w, np.abs(h)

    def __iter__(self):
        return (filter_ for filter_ in self.filter_bands)
    
    def __getitem__(self, index):
        return self.filter_bands[index]

    def __len__(self):
        return len(self.filter_bands)

    def process(self, signal):
        """
        Process the input signal through the filter chain.
        """
        output_signal = signal.copy()

        for filter_ in self.filter_bands:
            output_signal = lfilter(filter_.b, filter_.a, output_signal)
            
        return output_signal
