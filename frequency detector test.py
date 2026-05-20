# -*- coding: utf-8 -*-
"""
Created on Thu May  7 13:37:40 2026

@author: am7825
"""

import numpy as np
import sounddevice as sd

# Audio settings
SAMPLE_RATE = 48000
DURATION = 1.0  # seconds
#print(sd.query_devices(1))
# Musical note names

sd.default.device=(1,0)



def detect_frequency(audio, sample_rate):
    """
    Detect dominant frequency using FFT.
    """
    # Apply window to reduce noise
    windowed = audio * np.hanning(len(audio))

    # FFT
    fft = np.fft.rfft(windowed)
    frequencies = np.fft.rfftfreq(len(windowed), 1 / sample_rate)

    # Magnitude
    magnitude = np.abs(fft)

    # Ignore very low frequencies
    min_freq = 50
    max_freq = 10000

    valid = np.where((frequencies >= min_freq) &
                     (frequencies <= max_freq))

    frequencies = frequencies[valid]
    magnitude = magnitude[valid]

    peak_index = np.argmax(magnitude)
    dominant_freq = frequencies[peak_index]

    return dominant_freq


print("Listening... Press Ctrl+C to stop.")

try:
    while True:
        # Record audio
        audio = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float64'
        )

        sd.wait()

        # Flatten audio array
        audio = audio.flatten()

        # Detect frequency
        freq = detect_frequency(audio, SAMPLE_RATE)

        

        print(f"Frequency: {freq:.2f} Hz  ")

except KeyboardInterrupt:
    print("Stopped.")
#print(sd.default.device)
#print(sd.query_devices())

