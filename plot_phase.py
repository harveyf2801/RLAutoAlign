#! python3
#%%
import numpy as np
import librosa
from pathlib import Path
import matplotlib.pyplot as plt
from utilities import get_magnitude_and_phase_stft, auto_polarity_detection, unwrap

# Load input and target audio files
INPUT, FS = librosa.load(Path(f"soundfiles/KickStemIn.wav"), mono=True, sr=None)
TARGET, FS = librosa.load(Path(f"soundfiles/KickStemOut.wav"), mono=True, sr=None)

# Check the polarity of the audio files
POL_INVERT = auto_polarity_detection(INPUT, TARGET)
print("The polarity of the input signal",
      "needs" if POL_INVERT else "does not need",
      "to be inverted.")

#%%
# Compute the spectrogram for both audio files and
# separate magnitude and phase for both audio files

S_INPUT, phi_INPUT = get_magnitude_and_phase_stft(-INPUT if POL_INVERT else INPUT)
S_TARGET, phi_TARGET = get_magnitude_and_phase_stft(TARGET)

phi_INPUT = unwrap(phi_INPUT, dim=1)
phi_TARGET = unwrap(phi_TARGET, dim=1)

# Compute the phase difference between the two audio files
mag_sum = S_INPUT + S_TARGET
phi_diff = phi_INPUT - phi_TARGET

#%%
# Plot the phase difference
plt.figure(figsize=(10, 4))
librosa.display.specshow(phi_diff, y_axis='log', x_axis='time', sr=FS)
plt.colorbar()
plt.tight_layout()
plt.show()

#%%
# To visualize this more clearly, we can compute the time difference of unwrapped phase and plot that instead
delta_phi = np.diff(phi_diff, axis=1)

plt.figure(figsize=(10, 4))
librosa.display.specshow(delta_phi, y_axis='log', x_axis='time', sr=FS)
plt.title('unwrapped phase differential')
plt.colorbar()
plt.tight_layout()
plt.show()

#%%
# Plot phase differential and magnitude on a log-frequency scale
plt.figure(figsize=(10, 8))
# plt.subplot(2,1,1)
librosa.display.specshow(librosa.core.amplitude_to_db(mag_sum**2, ref=np.max), y_axis='log', sr=FS)
plt.title('log STFT power')
plt.colorbar()
plt.show()

# plt.subplot(2,1,2)
# librosa.display.specshow(librosa.core.amplitude_to_db(mag_sum_2**2, ref=np.max), y_axis='log', sr=FS)
# plt.title('log STFT power')
# plt.colorbar()

# #%%
# import sounddevice as sd
# sd.play((INPUT*0.5)+(TARGET*0.5), FS)
# sd.wait()

# #%%
# sd.play(((INPUT*-1)*0.5) + (TARGET*0.5), FS)
# sd.wait()
# %%
