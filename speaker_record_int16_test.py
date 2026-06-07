import numpy as np
import sounddevice as sd

SAMPLE_RATE = 48000
DURATION = 1
DEVICE_ID = 0

INPUT_CHANNELS = 2
OUTPUT_CHANNELS = 2

AMP = 10000 # same quiet-audible value from the aplay/int16 test

n = int(DURATION * SAMPLE_RATE)
t = np.arange(n) / SAMPLE_RATE

mono = (AMP * np.sin(2 * np.pi * 100 * t)).astype(np.int16)
output = np.column_stack([mono, mono]).astype(np.int16)

print(sd.query_devices(DEVICE_ID))
print("Output dtype:", output.dtype)
print("Output max:", np.max(np.abs(output)))

print("Playing quiet int16 sine and recording both mics...")

recording = sd.playrec(
    output,
    samplerate=SAMPLE_RATE,
    channels=INPUT_CHANNELS,
    device=DEVICE_ID,
    dtype="float32",
    blocking=True
)

print("Recording shape:", recording.shape)

for ch in range(INPUT_CHANNELS):
    x = recording[:, ch]
    print(f"\nChannel {ch}:")
    print("  Peak:", np.max(np.abs(x)))
    print("  RMS:", np.sqrt(np.mean(x ** 2)))
    print("  Power:", np.mean(x ** 2))
