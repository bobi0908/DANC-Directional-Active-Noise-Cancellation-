import numpy as np
from scipy.io import wavfile

SAMPLE_RATE = 48000
DURATION = 1

# 16-bit full scale is 32767.
# Start quiet.
AMP = 500

n = int(DURATION * SAMPLE_RATE)
t = np.arange(n) / SAMPLE_RATE

audio = (AMP * np.sin(2 * np.pi * 400 * t)).astype(np.int16)

wavfile.write("quiet_s16.wav", SAMPLE_RATE, audio)

print("Wrote quiet_s16.wav")
print("AMP:", AMP)
print("Max:", np.max(np.abs(audio)))
