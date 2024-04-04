#! python3
#%%
import numpy as np
import librosa
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Load input and target audio files
INPUT, FS = librosa.load(Path(f"soundfiles/KickStemIn.wav"), mono=True, sr=None)
TARGET, FS = librosa.load(Path(f"soundfiles/KickStemOut.wav"), mono=True, sr=None)

# Compute the spectrogram
D = librosa.stft(INPUT)

# Separate magnitude and phase
# librosa.magphase will do this, but it returns phase as exp(1.j * phi)
# for visualization, we just want phi itself
S, phi = np.abs(D), np.angle(D)

#%%
# First thing: plot the phase directly
# Not much obvious structure here
# We'll use a log-frequency axis scaling
plt.figure(figsize=(10, 4))
librosa.display.specshow(phi, y_axis='log', x_axis='time', sr=FS)
plt.colorbar()
plt.tight_layout()
plt.show()

#%%
# The representation above is difficult to understand visually for a couple of reasons:
#   1) phase wraps around +- pi, so red and blue are actually the same
#   2) frames overlap in time, so what matters isn't so much the phase at any point, but the way it evolves
#
# Unwrapping the phase in time can help with these issues, but we're not done yet
phi_unwrap = np.unwrap(phi, axis=1)

# Unwrap the phase along the time axis
plt.figure(figsize=(10, 4))
librosa.display.specshow(phi_unwrap, y_axis='log', x_axis='time', sr=FS)
plt.colorbar()
plt.tight_layout()
plt.show()

#%%
# If you pick any row of the unwrapped phase matrix, you see that it generally increases linearly in time,
# with a few discontinuities.  These are usually the interesting parts.
plt.figure(figsize=(10, 4))
plt.plot(phi_unwrap[30:40].T)
plt.legend(['{:.0f} Hz'.format(_) for _ in librosa.fft_frequencies(sr=FS)[30:40]],
           frameon=True, ncol=3, loc='best')
plt.axis('tight')
plt.xticks([])
plt.ylabel('$\phi$')
plt.tight_layout()
plt.show()

#%%
# To visualize this more clearly, we can compute the time difference of unwrapped phase and plot that instead
delta_phi = np.diff(phi_unwrap, axis=1)

# Only plot the time differential of phase
plt.figure(figsize=(10, 4))
librosa.display.specshow(delta_phi, y_axis='log', x_axis='time', sr=FS)
plt.colorbar()
plt.tight_layout()
plt.show()

# %%
# Plot phase differential and magnitude on a log-frequency scale
plt.figure(figsize=(10, 8))
plt.subplot(2,1,1)
librosa.display.specshow(librosa.core.logamplitude(S**2, ref_power=np.max), y_axis='log', sr=FS)
plt.title('log STFT power')
plt.colorbar()

plt.subplot(2,1,2)
librosa.display.specshow(delta_phi, y_axis='log', x_axis='time', sr=FS)
plt.title('unwrapped phase differential')
plt.colorbar()
plt.tight_layout()
plt.show()