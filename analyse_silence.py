import numpy as np
import sounddevice as sd

SAMPLE_RATE = 48000
DURATION = 10
DEVICE_ID = 0
CHANNELS = 2
IGNORE_START_SECONDS = 1

print("Recording silence. Do not touch anything.")

recording = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    device=DEVICE_ID,
    dtype="float32"
)

sd.wait()

# Remove startup transient
start = int(IGNORE_START_SECONDS * SAMPLE_RATE)
recording = recording[start:]

for ch in range(CHANNELS):
    x = recording[:, ch]
    print(f"\nChannel {ch}")
    print("  Peak:", np.max(np.abs(x)))
    print("  RMS:", np.sqrt(np.mean(x ** 2)))
    print("  Power:", np.mean(x ** 2))
    print("  Mean:", np.mean(x))

print("\nCorrelation:", np.corrcoef(recording[:, 0], recording[:, 1])[0, 1])

for ch in range(CHANNELS):
    x = recording[:, ch]
    idx = np.argmax(np.abs(x))
    print(
        f"Channel {ch} biggest remaining spike at sample {idx}, "
        f"time after trim {idx / SAMPLE_RATE:.4f}s, value {x[idx]}"
    )
