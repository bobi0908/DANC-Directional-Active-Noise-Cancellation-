import numpy as np
import sounddevice as sd
from offline_cancellation import OfflineAdaptiveFilter as off
# Audio settings
SAMPLE_RATE = 48000
DURATION = 1.0  # seconds
BLOCKSIZE = int(SAMPLE_RATE * DURATION)

#print(sd.query_devices())
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
    min_freq = 20
    max_freq = 10000

    valid = np.where((frequencies >= min_freq) &
                     (frequencies <= max_freq))

    frequencies = frequencies[valid]
    magnitude = magnitude[valid]

    peak_index = np.argmax(magnitude)
    dominant_freq = frequencies[peak_index]

    return dominant_freq


mic1 = sd.InputStream(device=19, channels=1, samplerate=SAMPLE_RATE)
mic2 = sd.InputStream(device=21, channels=1, samplerate=SAMPLE_RATE)

mic1.start()
mic2.start()

print("Listening... Ctrl+C to stop")

try:
    while True:
        # Read separately from each mic
        audio1, _ = mic1.read(BLOCKSIZE)
        audio2, _ = mic2.read(BLOCKSIZE)

        audio1 = audio1.flatten()
        audio2 = audio2.flatten()

        freq1 = detect_frequency(audio1, SAMPLE_RATE)
        freq2 = detect_frequency(audio2, SAMPLE_RATE)

        print(f"Mic1: {freq1:.2f} Hz | Mic2: {freq2:.2f} Hz")

except KeyboardInterrupt:
    print("Stopped")
    mic1.stop()
    mic2.stop()
    mic1.close()
    mic2.close()

