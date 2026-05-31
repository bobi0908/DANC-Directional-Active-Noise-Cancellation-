import numpy as np
from scipy.io import wavfile

sr, audio = wavfile.read("test.wav")

print("Sample rate:", sr)
print("Shape:", audio.shape)
print("Dtype:", audio.dtype)

audio = np.asarray(audio)

if np.issubdtype(audio.dtype, np.integer):
    audio_float = audio.astype(np.float32) / np.iinfo(audio.dtype).max
else:
    audio_float = audio.astype(np.float32)

print("Peak:", np.max(np.abs(audio_float)))
print("RMS:", np.sqrt(np.mean(audio_float ** 2)))
print("Power:", np.mean(audio_float ** 2))
print("First 20 samples:", audio[:20])
