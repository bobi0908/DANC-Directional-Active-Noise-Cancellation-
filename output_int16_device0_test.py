import numpy as np
import sounddevice as sd

SAMPLE_RATE = 48000
DURATION = 1
DEVICE_ID = 0

# Same logic as the WAV test.
# 16-bit full scale is 32767.
AMP = 500   # use the value that was quiet but audible with aplay

n = int(DURATION * SAMPLE_RATE)
t = np.arange(n) / SAMPLE_RATE

mono = (AMP * np.sin(2 * np.pi * 400 * t)).astype(np.int16)

# Device reports 2 output channels, so send the same signal to both.
output = np.column_stack([mono, mono]).astype(np.int16)

print(sd.query_devices(DEVICE_ID))
print("AMP =", AMP)
print("Output dtype:", output.dtype)
print("Output max:", np.max(np.abs(output)))

sd.play(
    output,
    samplerate=SAMPLE_RATE,
    device=DEVICE_ID,
    blocking=True
)

print("Done.")
