# Filter 0: frequency=11325.339367240667, q=9.214534482359886
# Filter 1: frequency=13792.853527069092, q=4.408732613921165
# Filter 2: frequency=18926.47318959236, q=2.9252729743719104
# Filter 3: frequency=14252.7046880126, q=4.9066004668362435
# Filter 4: frequency=5634.909717440605, q=0.24526146650314332
# Filter 5: frequency=18234.993906617165, q=5.004454688215628
# Filter 6: frequency=4348.827954530716, q=1.0386701315641405
# Filter 7: frequency=9399.653885886073, q=4.194673612713814
# Filter 8: frequency=18209.488384127617, q=2.5058218598365785
# Filter 9: frequency=924.7422260046005, q=4.499128475785255

filters_ = [
{'frequency': 11325.339367240667, 'q': 9.214534482359886},
{'frequency': 13792.853527069092, 'q': 4.408732613921165},
{'frequency': 18926.47318959236, 'q': 2.9252729743719104},
{'frequency': 14252.7046880126, 'q': 4.9066004668362435},
{'frequency': 5634.909717440605, 'q': 0.24526146650314332},
{'frequency': 18234.993906617165, 'q': 5.004454688215628},
{'frequency': 4348.827954530716, 'q': 1.0386701315641405},
{'frequency': 9399.653885886073, 'q': 4.194673612713814},
{'frequency': 18209.488384127617, 'q': 2.5058218598365785},
{'frequency': 924.7422260046005, 'q': 4.499128475785255}
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

    INPUT, FS = librosa.load(Path(f"/Users/s20113452/RLAutoAlign/soundfiles/SDDS_segmented_Allfiles/{input_snare}.wav"), mono=True, sr=None)
    TARGET, FS = librosa.load(Path(f"/Users/s20113452/RLAutoAlign/soundfiles/SDDS_segmented_Allfiles/{target_snare}.wav"), mono=True, sr=None)

    filter_chain = FilterChain([AllPassBand(filter_['frequency'], filter_['q'], FS) for filter_ in filters_])

    # Check the polarity of the audio files
    POL_INVERT = auto_polarity_detection(INPUT, TARGET)
    print("The polarity of the input signal",
        "needs" if POL_INVERT else "does not need",
        "to be inverted.")
    
    filtered_sig = filter_chain.process(-INPUT if POL_INVERT else INPUT)
    INPUT = -INPUT if POL_INVERT else INPUT

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

