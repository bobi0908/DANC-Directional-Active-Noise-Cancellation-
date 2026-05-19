import sounddevice as sd
import numpy as np
from preprocess import preprocess
from analyse_noise import analyse_noise

SAMPLE_RATE = 44100
DURATION = 3  # seconds

print("Recording real microphone audio for 3 seconds...")
audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype=np.float32
)

sd.wait()

print("Raw audio shape:", audio.shape)
print("Raw audio dtype:", audio.dtype)

processed = preprocess(audio)
noise_profile = analyse_noise(processed, SAMPLE_RATE)

frequencies = noise_profile["frequencies"]
magnitudes = noise_profile["magnitudes"]

peak_index = np.argmax(magnitudes)
peak_frequency = frequencies[peak_index]
peak_magnitude = magnitudes[peak_index]



#checking to see wag1 whats up 
print("Peak frequency:", peak_frequency)
print("Peak magnitude:", peak_magnitude)

print("Processed audio shape:", processed.shape)
print("Processed audio dtype:", processed.dtype)
print("Raw max:", np.max(np.abs(audio)))
print("Processed max:", np.max(np.abs(processed)))