import numpy as np
import librosa
import pywt

def auto_polarity_detection(signal, ref_signal):
    '''
    Auto polarity detection using decibel RMS level comparison.
    Returns True if polarity inversion is needed.
    '''
    # Reference value for decibel calculation
    reference = 1.0

    # Calculate the root mean square (RMS) decibel level of the signals before polarity inversion
    rms_before = np.sqrt(np.mean(signal + ref_signal ** 2))
    
    # Calculate decibel level
    db_before = 20 * np.log10(rms_before / reference)

    # Invert the polarity of signal
    # Calculate the RMS decibel level of the signals after polarity inversion
    power_after = np.sqrt(np.mean((-signal) + ref_signal ** 2))
    db_after = 20 * np.log10(power_after / reference)

    # Compare the decibel levels and return the polarity detection result
    return db_before >= db_after

def get_magnitude_and_phase_stft(signal,
        fft_size=1024,
        hop_size=256,
        win_length=1024,
        window="hann",
        eps=1e-8):
    '''
    Returns the magnitude and phase of the spectrogram of the audio signal.
    '''
    # Compute the spectrogram of the audio file
    x_stft = librosa.stft(signal,
                            n_fft=fft_size,
                            hop_length=hop_size,
                            window=window)

    # Separate magnitude and phase
    # For magnitude ...
    # (The np.abs function internally performs the same operation,
    # without clipping. This expression ensures values are no less than eps.)
    x_mag = np.sqrt(np.clip((np.real(x_stft) ** 2) + (np.imag(x_stft) ** 2), a_min=eps, a_max=None))
    x_phase = np.unwrap(np.angle(x_stft), axis=1)

    return x_mag, x_phase

def get_magnitude_and_phase_mrstft(signal,
        fft_sizes=[1024, 512, 2048],
        hop_sizes=[120, 50, 240],
        win_lengths=[600, 240, 1200],
        window="hann"):
    '''
    Returns the magnitude and phase of the multi-resolution spectrogram of the audio signal.
    '''
    assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)  # must define all

    pass
