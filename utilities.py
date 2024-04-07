import numpy as np
import librosa
import pywt

def auto_polarity_detection(signal1, signal2):
    '''
    Auto polarity detection using summed signal comparison.

    Method 1:
        Compare the summed signal before and after polarity inversion.
    '''
    # Calculate the sum of the signals before polarity inversion
    sum_before = np.sum(signal1 + signal2)

    # Invert the polarity of signal2
    # Calculate the sum of the signals after polarity inversion
    sum_after = np.sum(signal1 + (-signal2))

    # Compare the sums and return the polarity detection result
    return sum_before >= sum_after

def auto_polarity_detection2(signal1, signal2):
    '''
    Auto polarity detection using cross-correlation.

    Method 2:
        Compare the maximum cross-correlation before and after polarity inversion.
    '''
    # Calculate the cross-correlation of the signals before polarity inversion
    cross_correlation_before = np.correlate(signal1, signal2, mode='full')

    # Calculate the cross-correlation of the signals after polarity inversion
    cross_correlation_after = np.correlate(signal1, -signal2, mode='full')

    # Find the maximum value in the cross-correlation before and after polarity inversion
    max_cross_correlation_before = np.max(cross_correlation_before)
    max_cross_correlation_after = np.max(cross_correlation_after)

    # Compare the maximum cross-correlation values and return the polarity detection result
    return max_cross_correlation_before >= max_cross_correlation_after

def get_magnitude_and_phase(signal):
    '''
    Returns the magnitude and phase of the spectrogram of the audio signal.
    '''
    # Compute the spectrogram of the audio file
    D = librosa.stft(signal)

    # Separate magnitude and phase
    magnitude = np.abs(D)
    phase = np.unwrap(np.angle(D), axis=1)

    return magnitude, phase

def get_magnitude_and_phase_wavelet(signal):
    '''
    Returns the magnitude and phase of the Continuous Wavelet Transform (CWT) of the audio signal.
    '''
    # Compute the Continuous Wavelet Transform (CWT) of the audio file
    widths = np.arange(1, 31)
    cwtmatr, freqs = pywt.cwt(signal, widths, 'morl')

    # Separate magnitude and phase
    magnitude = np.abs(cwtmatr)
    phase = np.unwrap(np.angle(cwtmatr), axis=1)

    return magnitude, phase