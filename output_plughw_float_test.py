import numpy as np
import sounddevice as sd

SAMPLE_RATE = 48000
DURATION = 1

# Use the exact ALSA path that worked with speaker-test
DEVICE = "plughw:2,0"

OUTPUT_GAIN = 0.001

n = int(DURATION * SAMPLE_RATE)
t = np.arange(n) / SAMPLE_RATE

mono = OUTPUT_GAIN * np.sin(2 * np.pi * 400 * t).astype(np.float32)

# Try mono first, because speaker-test used -c1
output = mono.reshape(-1, 1)

print("Using device:", DEVICE)
print("Output shape:", output.shape)
print("Output max:", np.max(np.abs(output)))

sd.play(
    output,
    samplerate=SAMPLE_RATE,
    device=DEVICE,
    blocking=True
)

print("Done.")
