from abc import ABC, abstractmethod
import numpy as np
from scipy.signal import freqz


class FilterBand(ABC):
    """
    Filter Band Base Class for an audio EQ.
    """

    def __init__(self, filterName, frequency, q, dBgain, fs, order=2, warped=False):
        """
        Constructor for the FilterBand class.

        Parameters:
        - filterName: Name of the filter band
        - frequency: Center frequency of the filter (can range from 1Hz to nyquist fs/2)
        - q: Quality factor, which determines the shape of the filter at the center frequency (q can range from 0.1 to 40)
        - dBgain: Gain at the center frequency (in decibels)
        - fs: Sampling rate of the audio to apply the filter on
        - order: Order of the filter (default is 2)
        - warped: Boolean indicating if the filter is warped (default is False)
        """

        # Initialize filter coefficients
        self._a = np.ones(order + 1)
        self._b = self.a

        # Assign parameters
        self._filterName = filterName
        self._frequency = frequency
        self._q = q
        self._dBgain = dBgain
        self._fs = fs
        self._order = order
        self._warped = warped

        # Calculate initial values
        self._update_w0() # which then calls self._update_alpha()

        # Calculate filter coefficients
        self._calculate_coefficients()

        # All coefficients and parameters are dynamically updated as and when parameters are changed

    @property
    def filterName(self):
        return self._filterName

    @property
    def frequency(self):
        return self._frequency

    @frequency.setter
    def frequency(self, value):
        """
        Setter method for center frequency.
        """
        value = float(value)
        if 1 <= value <= self.fs / 2:
            self._frequency = value
            self._update_w0()
            self._calculate_coefficients()
        else:
            raise ValueError("Center frequency must be between 1Hz and nyquist fs/2")

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, value):
        """
        Setter method for quality factor.
        """
        value = float(value)
        if 0.1 <= value <= 40:
            self._q = value
            self._update_alpha()
            self._calculate_coefficients()
        else:
            raise ValueError("Quality factor must be between 0.1 and 40")

    @property
    def dBgain(self):
        return self._dBgain

    @dBgain.setter
    def dBgain(self, value):
        """
        Setter method for dB gain.
        """
        self._dBgain = float(value)
        self._calculate_coefficients()

    @property
    def fs(self):
        return self._fs

    @fs.setter
    def fs(self, value):
        """
        Setter method for sampling frequency.
        """
        self._fs = int(value)
        self._update_w0()
        self._calculate_coefficients()

    @property
    def order(self):
        # return max(len(self.a), len(self.b)) - 1
        return self._order

    # NOT IMPLEMENTED
    # @order.setter
    # def order(self, value):
    #     value = int(value)
    #     if value > 0:
    #         self._order = value
    #     else:
    #         raise ValueError("Filter order must be greater than 0")

    @property
    def warped(self):
        return self._warped
    
    # NOT IMPLEMENTED
    # @warped.setter
    # def warped(self, value):
    #     self._warped = bool(value)

    @property
    def w0(self):
        return self._w0
    
    def _update_w0(self):
        """
        Update angular frequency.
        """
        self._w0 = 2 * np.pi * self.frequency / self.fs
        self._update_alpha()
    
    @property
    def alpha(self):
        return self._alpha
    
    def _update_alpha(self):
        """
        Update alpha based on angular frequency and quality factor.
        """
        self._alpha = np.sin(self.w0) / (2 * self.q)

    @property
    def a(self):
        return self._a/self._a[0]

    @property
    def b(self):
        return self._b/self._a[0]

    @abstractmethod
    def _calculate_coefficients(self):
        """
        Abstract method to calculate filter coefficients.
        """
        raise NotImplementedError("Subclasses must implement calculate_coefficients method")
    
    def _get_response(self):
        return freqz(self.b, self.a, worN=8000, fs=self.fs)
    
    @property
    def phase_response(self):
        """
        Get the phase response of the filter.
        """
        w, h = self._get_response()
        return w, np.angle(h, deg=True)
    
    @property
    def magnitude_response(self):
        """
        Get the magnitude response of the filter.
        """
        w, h = self._get_response()
        return w, np.abs(h)
