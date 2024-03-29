import numpy as np

from Filters.FilterBand import FilterBand


class AllPassBand(FilterBand):
    """
    All-Pass Band Class for creating an all-pass filter band for an EQ.
    """

    def __init__(self, frequency, q, fs, order=2, warped=False):
        """
        Constructor for the AllPassBand class.

        Parameters:
        - frequency: Center frequency of the all-pass filter
                      (can range from 1Hz to nyquist fs/2)
        - q: Quality factor, which determines the shape of the filter
               at the center frequency (q can range from 0.1 to 40)
        - fs: Sampling rate of the audio to apply the filter on
        - order: Order of the filter (default is 2)
        - warped: Boolean indicating if the filter is warped (default is False)
        """
        super().__init__("All-Pass", frequency, q, 0, fs, order, warped)  # default setting the dBgain to 0

    def _calculate_coefficients(self):
        """
        CALCULATE COEFFICIENTS Outputs IIR coeffs b and a for a standard all-pass filter band.
        """

        # Calculating the filter coefficients
        
        # The following coeff pairs are equal:
        # b0 and a2, b1 and a1, b2 and a2

        b0_a2 = 1 - self.alpha
        b1_a1 = -2 * np.cos(self.w0)
        b2_a0 = 1 + self.alpha

        self._b = np.array([b0_a2, b1_a1, b2_a0])
        self._a = np.array([b2_a0, b1_a1, b0_a2])
