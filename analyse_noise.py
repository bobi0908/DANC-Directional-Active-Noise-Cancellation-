import sounddevice as sd
import numpy as np
import time

import sounddevice as sd
INPUT_CHANNELS = 1      

#analysing sound content 
#chunk at a time fr 
#now we have the clean chunk from preprocess ayyyy
def analyse_noise(audio_frame, sample_rate=44100):
    # Use first channel if audio is shaped like (samples, channels)
    if audio_frame.ndim == 2:
        audio_frame = audio_frame[:, 0]

    # Apply window to reduce FFT edge effects
    window = np.hanning(len(audio_frame))
    windowed_audio = audio_frame * window

    # Convert time-domain audio into frequency-domain
    spectrum = np.fft.rfft(windowed_audio)

    # Strength of each frequency
    magnitudes = np.abs(spectrum)

    # Frequency values for each FFT bin
    frequencies = np.fft.rfftfreq(len(audio_frame), d=1 / sample_rate)

    noise_profile = {
        "frequencies": frequencies,
        "magnitudes": magnitudes,
    }

    return noise_profile