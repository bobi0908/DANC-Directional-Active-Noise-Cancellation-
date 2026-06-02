import numpy as np
import sounddevice as sd

SAMPLE_RATE = 48000
DURATION = 0.5
DEVICE_ID = 0

INPUT_CHANNELS = 2
OUTPUT_CHANNELS = 1

OUTPUT_GAIN = 0.0001  # keep low at first

n_samples = int(DURATION * SAMPLE_RATE)
t = np.arange(n_samples) / SAMPLE_RATE

# Quiet 400 Hz sine wave
output_signal = OUTPUT_GAIN * np.sin(2 * np.pi * 400 * t).astype(np.float32)

# sounddevice wants output shape = (samples, output_channels)
output_signal = output_signal.reshape(-1, 1)

print("Using device:")
print(sd.query_devices(DEVICE_ID))

print("Playing quiet sine wave and recording both mics...")
print("OUTPUT_GAIN =", OUTPUT_GAIN)
print("Output max amplitude =", np.max(np.abs(output_signal)))
recording = sd.playrec(
    output_signal,
    samplerate=SAMPLE_RATE,
    channels=INPUT_CHANNELS,
    device=DEVICE_ID,
    dtype="float32"
)

sd.wait()

print("Recording shape:", recording.shape)

for ch in range(INPUT_CHANNELS):
    x = recording[:, ch]
    print(f"\nChannel {ch}:")
    print("  Peak:", np.max(np.abs(x)))
    print("  RMS:", np.sqrt(np.mean(x ** 2)))
    print("  Power:", np.mean(x ** 2))
