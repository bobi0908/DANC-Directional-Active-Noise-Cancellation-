import sounddevice as sd
import numpy as np

DEVICE = 0
SAMPLE_RATE = 48000
SECONDS = 3

print(sd.query_devices())

print("Recording for 3 seconds. Clap or speak loudly now.")
data = sd.rec(
    int(SECONDS * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=2,
    dtype="float32",
    device=DEVICE,
)
sd.wait()

print("shape:", data.shape)
print("left min/max:", np.min(data[:, 0]), np.max(data[:, 0]))
print("right min/max:", np.min(data[:, 1]), np.max(data[:, 1]))
print("left rms:", np.sqrt(np.mean(data[:, 0] ** 2)))
print("right rms:", np.sqrt(np.mean(data[:, 1] ** 2)))
print("first 20 samples left:", data[:20, 0])
print("first 20 samples right:", data[:20, 1])
