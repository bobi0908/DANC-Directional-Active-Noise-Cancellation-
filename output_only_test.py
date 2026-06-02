import numpy as np
import sounddevice as sd

SAMPLE_RATE = 48000
DURATION = 1
DEVICE_ID = 0

OUTPUT_GAIN = 1e-7

n = int(DURATION * SAMPLE_RATE)
t = np.arange(n) / SAMPLE_RATE

mono = OUTPUT_GAIN * np.sin(2 * np.pi * 400 * t).astype(np.float32)
output = np.column_stack([mono, mono]).astype(np.float32)

print(sd.query_devices(DEVICE_ID))
print("OUTPUT_GAIN =", OUTPUT_GAIN)
print("Output max amplitude =", np.max(np.abs(output)))

sd.play(output, samplerate=SAMPLE_RATE, device=DEVICE_ID, blocking=True)

print("Done.")
