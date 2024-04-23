import librosa
from pathlib import Path
from utilities import auto_polarity_detection, get_rms_decibels
from Filters.FilterChain import FilterChain
from Filters.AllPassBand import AllPassBand
import sounddevice as sd
from Visualisation import Visualisation
import numpy as np
from losses import MultiResolutionSTFTLoss

def stuff():
    FS = 44100

    # INPUT, FS = librosa.load(Path(f"soundfiles/KickStemIn.wav"), mono=True, sr=FS)
    # TARGET, FS = librosa.load(Path(f"soundfiles/KickStemOut.wav"), mono=True, sr=FS)

    # Create input sine wave
    duration = 5  # Duration in seconds
    frequency = 440  # Frequency in Hz
    amplitude = 0.5  # Amplitude of the sine wave
    phase = 0.8 # 0 more of of phase - 1 more in phase
    time = np.linspace(0, duration, int(FS * duration))
    INPUT = amplitude * np.sin(2 * np.pi * frequency * time + (np.pi))
    FILTERED = amplitude * np.sin(2 * np.pi * frequency * time + (np.pi*phase))
    TARGET = amplitude * np.sin(2 * np.pi * frequency * time)

    loss = MultiResolutionSTFTLoss()
    original_loss = loss.forward(INPUT, TARGET)
    filtered_loss = loss.forward(FILTERED, TARGET)
    print("Original Loss: ", original_loss)
    print("Filtered Loss: ", filtered_loss)

    # reward = original

    print("Reward: ", 100 - (original_loss / 510) * 100)
    print("Out of phase: ", ((np.pi*phase)/np.pi)*100)

    # Uncomment the following lines to load sine waves from sound files
    # INPUT, FS = librosa.load(Path(f"soundfiles/input.wav"), mono=True, sr=FS)
    # TARGET, FS = librosa.load(Path(f"soundfiles/target.wav"), mono=True, sr=FS)

    # filter_chain = FilterChain([AllPassBand(filter_['frequency'], filter_['q'], FS) for filter_ in filters_])

    # Check the polarity of the audio files
    # POL_INVERT = auto_polarity_detection(INPUT, TARGET)
    # print("The polarity of the input signal",
    #     "needs" if POL_INVERT else "does not need",
    #     "to be inverted.")

    # filtered_sig = filter_chain.process(-INPUT if POL_INVERT else INPUT)

    before_sig = (FILTERED*0.5) + (TARGET*0.5)
    # after_sig = (filtered_sig*0.5) + (TARGET*0.5)

    sd.play(before_sig)
    sd.wait()

    # sd.play(after_sig)
    # sd.wait()



original = -0.004

losses = [-510, -9, -0.003]

loss_range = (-510, 0)


for loss in losses:
    if loss >= original:
        # Positive impact
        # Rescale from original -> 0 to 0 -> 1
        reward = np.interp(loss, (original, 0), (0, 1))
    else:
        # Negative impact
        # Rescale from -510 -> original to -1 -> 0
        reward = np.interp(loss, (-510, original), (-1, 0))

    print(reward)

