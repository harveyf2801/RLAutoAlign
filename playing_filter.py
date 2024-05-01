import librosa
from pathlib import Path
from utilities import auto_polarity_detection, get_rms_decibels
from Filters.FilterChain import FilterChain
from Filters.AllPassBand import AllPassBand
import sounddevice as sd
from Visualisation import Visualisation
from losses import MultiResolutionSTFTLoss
import numpy as np
import time
from scipy.io import wavfile

# *** 1

# {'frequency': 16762.252733707428, 'q': 2.6483403116464617},
# {'frequency': 11175.475144907832, 'q': 6.769550976157189},
# {'frequency': 15265.843598246574, 'q': 0.24752356112003326},
# {'frequency': 7601.351917982101, 'q': 8.761303305625916},
# {'frequency': 9932.832419164479, 'q': 1.960881879925728},
# {'frequency': 3240.120148062706, 'q': 1.9038249909877778},
# {'frequency': 20000.0, 'q': 9.408942145109176},
# {'frequency': 20000.0, 'q': 10.0},
# {'frequency': 20.0, 'q': 10.0},
# {'frequency': 8436.414389163256, 'q': 10.0}

input_snare = "Gretsch_BigFatSnare_AKG_414_BTM_Segment_10_peak_0.066"
target_snare = "Gretsch_BigFatSnare_AKG_414_BTM_Segment_2_peak_0.051"

# *** 2

# {'frequency': 20.0, 'q': 4.451991780474782},
# {'frequency': 11853.68459239602, 'q': 6.027833990752697},
# {'frequency': 13802.013638317585, 'q': 7.349395722150803},
# {'frequency': 2986.217119693756, 'q': 7.3893202021718025},
# {'frequency': 20000.0, 'q': 4.282904301583767},
# {'frequency': 20000.0, 'q': 4.601222275942564},
# {'frequency': 1807.1979343891144, 'q': 0.1},
# {'frequency': 20000.0, 'q': 0.8715439140796661},
# {'frequency': 20.0, 'q': 4.260163199156523},
# {'frequency': 8033.644145727158, 'q': 0.1}

# input_snare = "wooden15_BigFatSnare_Sennheiser_MD421_TP_Segment_7_peak_0.029"
# target_snare = "wooden15_BigFatSnare_Shure_SM57_BTM_Segment_7_peak_0.035"

# 93.1 % better than original using RMS diff ^

# *** 3

# {'frequency': 20.0, 'q': 8.139125201106072},
# {'frequency': 8395.488338768482, 'q': 2.3806944221258166},
# {'frequency': 5525.17207890749, 'q': 7.568467503786087},
# {'frequency': 4453.808242082596, 'q': 0.1},
# {'frequency': 20.0, 'q': 8.160270637273788},
# {'frequency': 10394.066848605871, 'q': 0.1},
# {'frequency': 18215.41192471981, 'q': 7.563450297713279},
# {'frequency': 5107.044425308704, 'q': 1.6021160930395129},
# {'frequency': 20000.0, 'q': 3.495129629969597},
# {'frequency': 7905.559191703796, 'q': 0.1}

# input_snare = "YamahaMaple_BigFatSnare_Sennheiser_e614_TP_Segment_80_peak_0.163"
# target_snare = "YamahaMaple_BigFatSnare_AKG_414_BTM_Segment_80_peak_0.132"

filters_ = [
{'frequency': 20.0, 'q': 8.139125201106072},
{'frequency': 8395.488338768482, 'q': 2.3806944221258166},
{'frequency': 5525.17207890749, 'q': 7.568467503786087},
{'frequency': 4453.808242082596, 'q': 0.1},
{'frequency': 20.0, 'q': 8.160270637273788},
{'frequency': 10394.066848605871, 'q': 0.1},
{'frequency': 18215.41192471981, 'q': 7.563450297713279},
{'frequency': 5107.044425308704, 'q': 1.6021160930395129},
{'frequency': 20000.0, 'q': 3.495129629969597},
{'frequency': 7905.559191703796, 'q': 0.1}
]

def scale(arr):
    ''' Scale the audio signal to the range of a 16-bit integer '''
    return np.int16(arr * 32767)

# Test the environment
if __name__ == "__main__":

    input_snare = "YamahaMaple_BigFatSnare_Sennheiser_e614_TP_Segment_80_peak_0.163"
    target_snare = "YamahaMaple_BigFatSnare_AKG_414_BTM_Segment_80_peak_0.132"

    audio_path = "C:/Users/hfret/Downloads/SDDS"

    INPUT, FS = librosa.load(Path(audio_path, f"{input_snare}.wav"), mono=True, sr=44100, duration=0.5)
    TARGET, FS = librosa.load(Path(audio_path, f"{target_snare}.wav"), mono=True, sr=44100, duration=0.5)

    filter_chain = FilterChain([AllPassBand(filter_['frequency'], filter_['q'], FS) for filter_ in filters_])

    # Normalize the signals
    INPUT /= np.max(np.abs(INPUT))
    TARGET /= np.max(np.abs(TARGET))

    # Check the polarity of the audio files
    loss = MultiResolutionSTFTLoss()

    POL_INVERT = abs(float(loss(INPUT, TARGET))) > abs(float(loss(-INPUT, TARGET)))
    print("The polarity of the input signal",
        "needs" if POL_INVERT else "does not need",
        "to be inverted.")
    
    INPUT = -INPUT if POL_INVERT else INPUT
    
    filtered_sig = filter_chain.process(INPUT)

    visuals = Visualisation("APF Response", 'graph_filters', FS, 5)
    visuals.show_mag = True
    visuals.render(filter_chain)

    before_sig = (INPUT + TARGET) / 2
    after_sig = (filtered_sig + TARGET) / 2

    loss1 = abs(float(loss(INPUT, TARGET)))
    loss2 = abs(float(loss(filtered_sig, TARGET)))
    print("Reward loss: ", (loss1-loss2))

    sd.play(before_sig)
    sd.wait()

    time.sleep(1)

    sd.play(after_sig)
    sd.wait()

    mix1_path = Path(f'model_snare1.wav')
    wavfile.write(mix1_path, FS, scale(after_sig))