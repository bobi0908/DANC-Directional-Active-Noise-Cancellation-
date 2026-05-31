import numpy as np
import sounddevice as sd
from offline_cancellation import OfflineAdaptiveFilter as off

SAMPLE_RATE = 48000
BLOCKSIZE = 1024

anc = off(filter_length=128, learning_rate=0.0001)


def make_1d(signal):
    """
    Convert sounddevice input shape (N, 1) into clean shape (N,).
    """
    return np.asarray(signal, dtype=np.float32).ravel()


def safe_corrcoef(a, b):
    """
    Correlation between two 1D signals, safely.
    """
    a = make_1d(a)
    b = make_1d(b)

    length = min(len(a), len(b))
    a = a[:length]
    b = b[:length]

    if length < 2:
        return np.nan

    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan

    return np.corrcoef(a, b)[0, 1]


def max_shifted_correlation(reference_signal, target_signal, max_shift=100):
    """
    Try different sample shifts and return the best correlation.
    """
    reference_signal = make_1d(reference_signal)
    target_signal = make_1d(target_signal)

    length = min(len(reference_signal), len(target_signal))
    reference_signal = reference_signal[:length]
    target_signal = target_signal[:length]

    corrs = []

    for shift in range(-max_shift, max_shift + 1):
        if shift < 0:
            ref = reference_signal[:shift]
            tgt = target_signal[-shift:]
        elif shift > 0:
            ref = reference_signal[shift:]
            tgt = target_signal[:-shift]
        else:
            ref = reference_signal
            tgt = target_signal

        c = safe_corrcoef(ref, tgt)
        corrs.append(c)

    return np.nanmax(corrs)


mic1 = sd.InputStream(
    device=4,
    channels=1,
    samplerate=SAMPLE_RATE,
    blocksize=BLOCKSIZE,
    dtype="float32"
)

mic2 = sd.InputStream(
    device=1,
    channels=1,
    samplerate=SAMPLE_RATE,
    blocksize=BLOCKSIZE,
    dtype="float32"
)

last_reference_signal = None
last_target_signal = None
last_error_signal = None

print("Reading audio... Press Ctrl+C to stop.")

try:
    mic1.start()
    mic2.start()

    while True:
        reference_signal, _ = mic1.read(BLOCKSIZE)
        target_signal, _ = mic2.read(BLOCKSIZE)

        reference_signal = make_1d(reference_signal)
        target_signal = make_1d(target_signal)

        # Make sure both blocks have the same length
        length = min(len(reference_signal), len(target_signal))
        reference_signal = reference_signal[:length]
        target_signal = target_signal[:length]

        predicted_signal = np.zeros(length, dtype=np.float32)
        error_signal = np.zeros(length, dtype=np.float32)

        for i in range(length):
            predicted, error = anc.process_sample(
                float(reference_signal[i]),
                float(target_signal[i])
            )

            predicted_signal[i] = predicted
            error_signal[i] = error

        last_reference_signal = reference_signal
        last_target_signal = target_signal
        last_error_signal = error_signal

except KeyboardInterrupt:
    print("Stopping...")

finally:
    mic1.stop()
    mic2.stop()
    mic1.close()
    mic2.close()
    print("Closed successfully")

best_lag = 1770 #change this every time you change the configuration of the mics

def align_signals(reference_signal, target_signal, lag):
    reference_signal = np.asarray(reference_signal).ravel()
    target_signal = np.asarray(target_signal).ravel()

    if lag > 0:
        # target is delayed relative to reference
        reference_aligned = reference_signal[:-lag]
        target_aligned = target_signal[lag:]

    elif lag < 0:
        # reference is delayed relative to target
        reference_aligned = reference_signal[-lag:]
        target_aligned = target_signal[:lag]

    else:
        reference_aligned = reference_signal
        target_aligned = target_signal

    length = min(len(reference_aligned), len(target_aligned))

    return reference_aligned[:length], target_aligned[:length]
reference_aligned, target_aligned = align_signals(
    reference_signal,
    target_signal,
    best_lag
)

if last_reference_signal is not None:
    print(f"Mean absolute error: {np.mean(np.abs(last_error_signal)):.8f}")
    print(f"Mean target signal power: {np.mean(last_target_signal ** 2):.8f}")
    print(f"Mean reference signal power: {np.mean(last_reference_signal ** 2):.8f}")
    print(f"Correlation: {safe_corrcoef(last_reference_signal, last_target_signal):.4f}")
    print(f"Best shifted correlation: {max_shifted_correlation(last_reference_signal, last_target_signal):.4f}")
else:
    print("No audio was recorded.")