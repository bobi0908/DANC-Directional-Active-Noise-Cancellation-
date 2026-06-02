import numpy as np
import sounddevice as sd

SAMPLE_RATE = 48000
DURATION = 0.3
DEVICE_ID = 0

# Start absurdly tiny in int32 units.
# Full-scale int32 is about 2.1 billion.
# 1000 is basically microscopic.
INT_AMPLITUDE =  2000000000

n = int(DURATION * SAMPLE_RATE)
t = np.arange(n) / SAMPLE_RATE

mono = (INT_AMPLITUDE * np.sin(2 * np.pi * 400 * t)).astype(np.int32)

# Device has 2 output channels
output = np.column_stack([mono, mono]).astype(np.int32)

print(sd.query_devices(DEVICE_ID))
print("INT_AMPLITUDE =", INT_AMPLITUDE)
print("Output max integer =", np.max(np.abs(output)))

sd.play(
    output,
    samplerate=SAMPLE_RATE,
    device=DEVICE_ID,
    blocking=True
)

print("Done.")
