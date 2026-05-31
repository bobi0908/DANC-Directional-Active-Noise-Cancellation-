import numpy as np
import sounddevice as sd
from scipy.signal import correlate, correlation_lags

SAMPLE_RATE = 48000
DURATION = 5.0
BLOCKSIZE = int(SAMPLE_RATE * DURATION)

DEVICE_1 = 4   # phone mic
DEVICE_2 = 1   # USB mic


def normalise(x):
    x = np.asarray(x, dtype=np.float32).ravel()
    x = x - np.mean(x)

    std = np.std(x)
    if std > 0:
        x = x / std

    return x


def record_from_device(device_id, duration, sample_rate):
    stream = sd.InputStream(
        device=device_id,
        channels=1,
        samplerate=sample_rate,
        dtype="float32"
    )

    with stream:
        audio, _ = stream.read(int(duration * sample_rate))

    return audio.ravel()


print("Starting streams...")

mic1 = sd.InputStream(
    device=DEVICE_1,
    channels=1,
    samplerate=SAMPLE_RATE,
    dtype="float32"
)

mic2 = sd.InputStream(
    device=DEVICE_2,
    channels=1,
    samplerate=SAMPLE_RATE,
    dtype="float32"
)

mic1.start()
mic2.start()

print("Recording... make a loud clap or play a steady tone now.")

audio1, _ = mic1.read(BLOCKSIZE)
audio2, _ = mic2.read(BLOCKSIZE)

mic1.stop()
mic2.stop()
mic1.close()
mic2.close()

x = normalise(audio1)
y = normalise(audio2)

# Cross-correlate full recordings
corr = correlate(y, x, mode="full")
lags = correlation_lags(len(y), len(x), mode="full")

best_index = np.argmax(np.abs(corr))
best_lag = lags[best_index]

# Normalised approximate correlation
best_corr = corr[best_index] / len(x)



print(f"Best lag: {best_lag} samples")
print(f"Best lag in seconds: {best_lag / SAMPLE_RATE:.4f} s")
print(f"Best absolute correlation: {abs(best_corr):.4f}")
print(f"Signed correlation: {best_corr:.4f}")