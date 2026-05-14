import sounddevice as sd
import numpy as np
from preprocess import preprocess

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

print("Processed audio shape:", processed.shape)
print("Processed audio dtype:", processed.dtype)
print("Raw max:", np.max(np.abs(audio)))
print("Processed max:", np.max(np.abs(processed)))