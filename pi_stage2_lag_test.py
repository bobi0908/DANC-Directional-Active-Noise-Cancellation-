import numpy as np
import sounddevice as sd
from scipy.signal import butter, sosfiltfilt, correlate, correlation_lags

SAMPLE_RATE = 48000
DURATION = 20
DEVICE_ID = 0  # change to your input device
CHANNELS = 2

REFERENCE_CHANNEL = 0
TARGET_CHANNEL = 1

LOWCUT = 80
HIGHCUT = 1000
IGNORE_START_SECONDS = 2


def make_1d(x):
    return np.asarray(x, dtype=np.float32).ravel()


def bandpass(x, sample_rate, low=80, high=1000, order=4):
    x = make_1d(x)
    nyquist = sample_rate / 2

    sos = butter(
        order,
        [low / nyquist, high / nyquist],
        btype="band",
        output="sos"
    )

    return sosfiltfilt(sos, x)


def normalise(x):
    x = make_1d(x)
    x = x - np.mean(x)
    std = np.std(x)
    if std > 0:
        x = x / std
    return x


def find_best_lag(reference_signal, target_signal, max_lag_seconds=0.1):
    ref = normalise(reference_signal)
    tgt = normalise(target_signal)

    corr = correlate(tgt, ref, mode="full")
    lags = correlation_lags(len(tgt), len(ref), mode="full")

    max_lag_samples = int(max_lag_seconds * SAMPLE_RATE)
    valid = np.abs(lags) <= max_lag_samples

    corr = corr[valid]
    lags = lags[valid]

    best_i = np.argmax(np.abs(corr))
    best_lag = int(lags[best_i])
    best_corr = corr[best_i] / len(ref)

    return best_lag, best_corr


def align_signals(reference_signal, target_signal, lag):
    reference_signal = make_1d(reference_signal)
    target_signal = make_1d(target_signal)

    if lag > 0:
        reference_aligned = reference_signal[:-lag]
        target_aligned = target_signal[lag:]
    elif lag < 0:
        reference_aligned = reference_signal[-lag:]
        target_aligned = target_signal[:lag]
    else:
        reference_aligned = reference_signal
        target_aligned = target_signal

    length = min(len(reference_aligned), len(target_aligned))
    return reference_aligned[:length], target_aligned[:length]


def print_levels(name, x):
    x = make_1d(x)
    print(f"{name}: peak={np.max(np.abs(x)):.6f}, rms={np.sqrt(np.mean(x**2)):.6f}, power={np.mean(x**2):.8e}")


print("Recording. Play pink noise / low-frequency noise near both mics.")

recording = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    device=DEVICE_ID,
    dtype="float32"
)

sd.wait()

reference = recording[:, REFERENCE_CHANNEL]
target = recording[:, TARGET_CHANNEL]

print_levels("Reference raw", reference)
print_levels("Target raw", target)

start = int(IGNORE_START_SECONDS * SAMPLE_RATE)

reference = reference[start:]
target = target[start:]

reference_low = bandpass(reference, SAMPLE_RATE, LOWCUT, HIGHCUT)
target_low = bandpass(target, SAMPLE_RATE, LOWCUT, HIGHCUT)

lag, lag_corr = find_best_lag(reference_low, target_low)

reference_aligned, target_aligned = align_signals(reference_low, target_low, lag)

aligned_corr = np.corrcoef(
    normalise(reference_aligned),
    normalise(target_aligned)
)[0, 1]

print("\nLag test:")
print(f"Best lag: {lag} samples")
print(f"Best lag: {lag / SAMPLE_RATE:.5f} seconds")
print(f"Correlation at lag: {lag_corr:.4f}")
print(f"Aligned low-frequency correlation: {aligned_corr:.4f}")

print_levels("Reference aligned", reference_aligned)
print_levels("Target aligned", target_aligned)
