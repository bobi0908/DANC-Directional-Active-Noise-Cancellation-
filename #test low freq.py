#test low freq
import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.signal import correlate, correlation_lags
import sounddevice as sd



SAMPLE_RATE = 48000
DURATION = 20
DEVICE_ID = 7

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


def find_best_lag(x, y, sample_rate, max_lag_seconds=0.1):
    x = normalise(x)
    y = normalise(y)

    corr = correlate(y, x, mode="full")
    lags = correlation_lags(len(y), len(x), mode="full")

    max_lag_samples = int(max_lag_seconds * sample_rate)
    valid = np.abs(lags) <= max_lag_samples

    corr = corr[valid]
    lags = lags[valid]

    best_i = np.argmax(np.abs(corr))
    best_lag = lags[best_i]
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

# Apply this to your recorded channels:
mic_a_low = bandpass(mic_a, SAMPLE_RATE, low=80, high=1000)
mic_b_low = bandpass(mic_b, SAMPLE_RATE, low=80, high=1000)

lag, corr = find_best_lag(mic_a_low, mic_b_low, SAMPLE_RATE)

print(f"Low-frequency lag: {lag} samples")
print(f"Low-frequency lag: {lag / SAMPLE_RATE:.5f} s")
print(f"Low-frequency correlation: {corr:.4f}")
def align_signals(x, y, lag):
    x = make_1d(x)
    y = make_1d(y)

    if lag > 0:
        # y is delayed relative to x
        x_aligned = x[:-lag]
        y_aligned = y[lag:]
    elif lag < 0:
        # x is delayed relative to y
        x_aligned = x[-lag:]
        y_aligned = y[:lag]
    else:
        x_aligned = x
        y_aligned = y

    length = min(len(x_aligned), len(y_aligned))
    return x_aligned[:length], y_aligned[:length]


x_aligned, y_aligned = align_signals(mic_a_low, mic_b_low, lag)

actual_corr = np.corrcoef(
    normalise(x_aligned),
    normalise(y_aligned)
)[0, 1]

#print("Mic A peak:", np.max(np.abs(mic_a)))
#print("Mic B peak:", np.max(np.abs(mic_b)))
#print("Mic A power:", np.mean(mic_a ** 2))
#print("Mic B power:", np.mean(mic_b ** 2))
print("Actual aligned low-frequency correlation:", actual_corr)


def align_signals(x, y, lag):
    x = np.asarray(x, dtype=np.float32).ravel()
    y = np.asarray(y, dtype=np.float32).ravel()

    if lag > 0:
        x_aligned = x[:-lag]
        y_aligned = y[lag:]
    elif lag < 0:
        x_aligned = x[-lag:]
        y_aligned = y[:lag]
    else:
        x_aligned = x
        y_aligned = y

    length = min(len(x_aligned), len(y_aligned))
    return x_aligned[:length], y_aligned[:length]


def corr(a, b):
    a = normalise(a)
    b = normalise(b)

    length = min(len(a), len(b))
    return np.corrcoef(a[:length], b[:length])[0, 1]


# Ignore first 2 seconds
start = 2 * SAMPLE_RATE
mic_a_test = mic_a[start:]
mic_b_test = mic_b[start:]

# Bandpass
mic_a_low = bandpass(mic_a_test, SAMPLE_RATE, low=80, high=1000)
mic_b_low = bandpass(mic_b_test, SAMPLE_RATE, low=80, high=1000)

# Find lag on low-frequency signals
lag, _ = find_best_lag(mic_a_low, mic_b_low, SAMPLE_RATE, max_lag_seconds=0.05)

# Align the SAME low-frequency signals
a_low_aligned, b_low_aligned = align_signals(mic_a_low, mic_b_low, lag)

print(f"Lag: {lag} samples")
print(f"Lag seconds: {lag / SAMPLE_RATE:.5f}")
print(f"Low-frequency aligned correlation: {corr(a_low_aligned, b_low_aligned):.4f}")

print("Mic A peak:", np.max(np.abs(mic_a)))
print("Mic B peak:", np.max(np.abs(mic_b)))
print("Mic A power:", np.mean(mic_a ** 2))
print("Mic B power:", np.mean(mic_b ** 2))