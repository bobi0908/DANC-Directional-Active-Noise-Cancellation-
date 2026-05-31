import numpy as np
import sounddevice as sd
from scipy.signal import correlate, correlation_lags

SAMPLE_RATE = 48000
DURATION = 20
DEVICE_ID = 7 # change this to your Aggregate Device ID


def normalise(x):
    x = np.asarray(x, dtype=np.float32).ravel()
    x = x - np.mean(x)
    std = np.std(x)
    if std > 0:
        x = x / std
    return x


def find_best_lag(x, y, max_lag_seconds=1.0):
    x = normalise(x)
    y = normalise(y)

    corr = correlate(y, x, mode="full")
    lags = correlation_lags(len(y), len(x), mode="full")

    max_lag_samples = int(max_lag_seconds * SAMPLE_RATE)
    valid = np.abs(lags) <= max_lag_samples

    corr = corr[valid]
    lags = lags[valid]

    best_i = np.argmax(np.abs(corr))
    best_lag = lags[best_i]

    # Normalised correlation estimate
    best_corr = corr[best_i] / len(x)

    return best_lag, best_corr


print("Recording. Play white noise or pink noise near both mics.")
recording = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=2,
    device=DEVICE_ID,
    dtype="float32"
)
sd.wait()

mic_a = recording[:, 0]
mic_b = recording[:, 1]

print("Recording shape:", recording.shape)
print("Mic A power:", np.mean(mic_a ** 2))
print("Mic B power:", np.mean(mic_b ** 2))
print(np.max(np.abs(mic_a)))
print(np.max(np.abs(mic_b)))
segment_length = 5 * SAMPLE_RATE

segments = {
    "first 5s": (0, segment_length),
    "middle 5s": ((len(mic_a) // 2) - segment_length // 2,
                  (len(mic_a) // 2) + segment_length // 2),
    "last 5s": (len(mic_a) - segment_length, len(mic_a)),
}

for name, (start, end) in segments.items():
    lag, corr = find_best_lag(mic_a[start:end], mic_b[start:end], max_lag_seconds=0.2)
    print(f"{name}: lag = {lag} samples, {lag / SAMPLE_RATE:.5f} s, corr = {corr:.4f}")