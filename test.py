# Filter 0: frequency=119.94433004914043, q=2.0563879963594
# Filter 1: frequency=381.9943935728577, q=8.452033827496694
# Filter 2: frequency=71.65568778735842, q=0.7803688164440915
# Filter 3: frequency=528.5787750903811, q=9.068195883400865
# Filter 4: frequency=433.57931550220536, q=8.58292949477177

filters_ = [
    {'frequency': 119.94433004914043, 'q': 2.0563879963594},
    {'frequency': 381.9943935728577,  'q': 8.452033827496694},
    {'frequency': 71.65568778735842,  'q': 0.7803688164440915},
    {'frequency': 528.5787750903811,  'q': 9.068195883400865},
    {'frequency': 433.57931550220536, 'q': 8.58292949477177}
]

# Test the environment
if __name__ == "__main__":

    import librosa
    from pathlib import Path
    from utilities import auto_polarity_detection
    from Filters.FilterChain import FilterChain
    from Filters.AllPassBand import AllPassBand
    import sounddevice as sd
    from Visualisation import Visualisation

    INPUT, FS = librosa.load(Path(f"soundfiles/KickStemIn.wav"), mono=True, sr=None)
    TARGET, FS = librosa.load(Path(f"soundfiles/KickStemOut.wav"), mono=True, sr=None)

    filter_chain = FilterChain([AllPassBand(filter_['frequency'], filter_['q'], FS) for filter_ in filters_])

    # Check the polarity of the audio files
    POL_INVERT = auto_polarity_detection(INPUT, TARGET)
    print("The polarity of the input signal",
        "needs" if POL_INVERT else "does not need",
        "to be inverted.")
    
    filtered_sig = filter_chain.process(-INPUT if POL_INVERT else INPUT)

    visuals = Visualisation("APF Response", 'graph_filters', FS, 5)
    visuals.show_mag = True
    visuals.render(filter_chain)

    sd.play((INPUT*0.5) + (TARGET*0.5))
    sd.wait()

    sd.play((filtered_sig*0.5) + (TARGET*0.5))
    sd.wait()

