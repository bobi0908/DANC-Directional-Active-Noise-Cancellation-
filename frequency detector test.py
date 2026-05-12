# -*- coding: utf-8 -*-
"""
Created on Thu May  7 13:37:40 2026

@author: am7825
"""

import numpy as np
import sounddevice as sd

# Audio settings
SAMPLE_RATE = 44100
DURATION = 1.0  # seconds

# Musical note names
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
              'F#', 'G', 'G#', 'A', 'A#', 'B']


def frequency_to_note(freq):
    """
    Convert frequency to nearest musical note.
    A4 = 440 Hz
    """
    if freq <= 0:
        return None

    # MIDI note number
    midi = int(round(69 + 12 * np.log2(freq / 440.0)))

    note_name = NOTE_NAMES[midi % 12]
    octave = (midi // 12) - 1

    return f"{note_name}{octave}"


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

        # Convert to note
        note = frequency_to_note(freq)

        print(f"Frequency: {freq:.2f} Hz  ->  Note: {note}")

except KeyboardInterrupt:
    print("Stopped.")