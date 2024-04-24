# Filter 0: frequency=20.0, q=4.451991780474782
# Filter 1: frequency=11853.68459239602, q=6.027833990752697
# Filter 2: frequency=13802.013638317585, q=7.349395722150803
# Filter 3: frequency=2986.217119693756, q=7.3893202021718025
# Filter 4: frequency=20000.0, q=4.282904301583767
# Filter 5: frequency=20000.0, q=4.601222275942564
# Filter 6: frequency=1807.1979343891144, q=0.1
# Filter 7: frequency=20000.0, q=0.8715439140796661
# Filter 8: frequency=20.0, q=4.260163199156523
# Filter 9: frequency=8033.644145727158, q=0.1

# wooden15_BigFatSnare_Sennheiser_MD421_TP_Segment_7_peak_0.029
# wooden15_BigFatSnare_Shure_SM57_BTM_Segment_7_peak_0.035

# 93.1 % better than original

filters_ = [
{'frequency': 20.0, 'q': 4.451991780474782},
{'frequency': 11853.68459239602, 'q': 6.027833990752697},
{'frequency': 13802.013638317585, 'q': 7.349395722150803},
{'frequency': 2986.217119693756, 'q': 7.3893202021718025},
{'frequency': 20000.0, 'q': 4.282904301583767},
{'frequency': 20000.0, 'q': 4.601222275942564},
{'frequency': 1807.1979343891144, 'q': 0.1},
{'frequency': 20000.0, 'q': 0.8715439140796661},
{'frequency': 20.0, 'q': 4.260163199156523},
{'frequency': 8033.644145727158, 'q': 0.1}
]

# Test the environment
if __name__ == "__main__":

    import librosa
    from pathlib import Path
    from utilities import auto_polarity_detection, get_rms_decibels
    from Filters.FilterChain import FilterChain
    from Filters.AllPassBand import AllPassBand
    import sounddevice as sd
    from Visualisation import Visualisation

    input_snare = "Tama6.5_BigFatSnare_Audix_i5_BTM_Segment_37_peak_0.572"
    target_snare = "Tama6.5_BigFatSnare_Shure_SM7B_TP_Segment_37_peak_0.381"

    audio_path = "./soundfiles/SDDS_segmented_Allfiles"

    INPUT, FS = librosa.load(Path(audio_path, f"{input_snare}.wav"), mono=True, sr=None)
    TARGET, FS = librosa.load(Path(audio_path, f"{target_snare}.wav"), mono=True, sr=None)

    filter_chain = FilterChain([AllPassBand(filter_['frequency'], filter_['q'], FS) for filter_ in filters_])

    # Check the polarity of the audio files
    from losses import MultiResolutionSTFTLoss
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

    before_sig = (INPUT*0.5) + (TARGET*0.5)
    after_sig = (filtered_sig*0.5) + (TARGET*0.5)

    print(get_rms_decibels(filtered_sig) - get_rms_decibels(INPUT))
    print(get_rms_decibels(after_sig), get_rms_decibels(before_sig))

    sd.play(before_sig)
    sd.wait()

    sd.play(after_sig)
    sd.wait()

