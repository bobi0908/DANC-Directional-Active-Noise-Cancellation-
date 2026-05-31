import numpy as np
import sounddevice as sd
from scipy.signal import butter, sosfiltfilt
from offline_cancellation import OfflineNLMSFilter as off


# =========================================================
# SETTINGS
# =========================================================

SAMPLE_RATE = 48000
DURATION = 20
DEVICE_ID = 0
FIXED_LAG = 0    # use your calibrated lag, around 1768-1775

LOWCUT = 80
HIGHCUT = 1000
channels=2
device=DEVICE_ID
FILTER_LENGTH = 128
LEARNING_RATE = 0.0001


# =========================================================
# HELPERS
# =========================================================

def make_1d(x):
    return np.asarray(x, dtype=np.float32).ravel()



def compute_erle(target_signal, error_signal):
    target_power = np.mean(target_signal ** 2)
    error_power = np.mean(error_signal ** 2)
    return 10 * np.log10(target_power / error_power)


def test_params(reference_aligned, target_aligned):
    learning_rates = [0.02, 0.05]
    filter_lengths = [256, 512]

    for filter_length in filter_lengths:
        for lr in learning_rates:
            anc = off(filter_length=filter_length, learning_rate=lr)

            predicted_signal, error_signal = anc.process(
                reference_aligned,
                target_aligned
            )

            erle = compute_erle(target_aligned, error_signal)

            half = len(target_aligned) // 2
            late_erle = compute_erle(
                target_aligned[half:],
                error_signal[half:]
            )

            predicted_power = np.mean(predicted_signal ** 2)
            error_power = np.mean(error_signal ** 2)

            print(
                f"filter={filter_length:3d}, "
                f"lr={lr:.3f}, "
                f"ERLE={erle:7.2f} dB, "
                f"late ERLE={late_erle:7.2f} dB, "
                f"pred power={predicted_power:.3f}, "
                f"error power={error_power:.3f}"
            )


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


def align_signals(reference_signal, target_signal, lag):
    """
    Align target_signal relative to reference_signal.

    Positive lag means target is delayed relative to reference.
    """
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


def run_adaptive_filter(reference_signal, target_signal, filter_length=128, learning_rate=0.0001):
    """
    Runs the offline adaptive filter and returns predicted signal and error signal.
    """
    anc = off(filter_length=filter_length, learning_rate=learning_rate)

    # If your class has a .process() method, use it.
    if hasattr(anc, "process"):
        predicted_signal, error_signal = anc.process(reference_signal, target_signal)
        return predicted_signal, error_signal

    # Otherwise, fall back to process_sample().
    predicted_signal = np.zeros_like(reference_signal, dtype=np.float32)
    error_signal = np.zeros_like(reference_signal, dtype=np.float32)

    for i in range(len(reference_signal)):
        predicted, error = anc.process_sample(
            float(reference_signal[i]),
            float(target_signal[i])
        )

        predicted_signal[i] = predicted
        error_signal[i] = error

    return predicted_signal, error_signal

def rms_normalise(x):
    x = np.asarray(x, dtype=np.float32).ravel()
    rms = np.sqrt(np.mean(x ** 2))
    if rms > 0:
        return x / rms
    return x


def compute_error_metrics(target_signal, predicted_signal, error_signal):
    """
    Compare adaptive filter error against the baseline error.

    Baseline = predicting zero.
    So baseline error is just the target signal itself.
    """
    target_signal = make_1d(target_signal)
    predicted_signal = make_1d(predicted_signal)
    error_signal = make_1d(error_signal)

    baseline_power = np.mean(target_signal ** 2)
    predicted_power = np.mean(predicted_signal ** 2)
    error_power = np.mean(error_signal ** 2)

    normalised_error = error_power / baseline_power
    erle_db = 10 * np.log10(baseline_power / error_power)

    correlation_predicted_target = np.corrcoef(predicted_signal, target_signal)[0, 1]
    correlation_error_target = np.corrcoef(error_signal, target_signal)[0, 1]

    return {
        "baseline_target_power": baseline_power,
        "predicted_signal_power": predicted_power,
        "adaptive_error_power": error_power,
        "normalised_error": normalised_error,
        "erle_db": erle_db,
        "correlation_predicted_target": correlation_predicted_target,
        "correlation_error_target": correlation_error_target,
    }


def compute_late_stage_erle(target_signal, error_signal, fraction=0.5):
    """
    Computes ERLE only on the later part of the recording.

    This is useful because the first part includes the filter learning.
    """
    target_signal = make_1d(target_signal)
    error_signal = make_1d(error_signal)

    start = int(len(target_signal) * fraction)

    target_late = target_signal[start:]
    error_late = error_signal[start:]

    target_power_late = np.mean(target_late ** 2)
    error_power_late = np.mean(error_late ** 2)

    erle_late_db = 10 * np.log10(target_power_late / error_power_late)
    normalised_error_late = error_power_late / target_power_late

    return erle_late_db, normalised_error_late


def print_audio_levels(name, x):
    x = make_1d(x)

    peak = np.max(np.abs(x))
    rms = np.sqrt(np.mean(x ** 2))
    power = np.mean(x ** 2)

    print(f"{name}:")
    print(f"  Peak:  {peak:.6f}")
    print(f"  RMS:   {rms:.6f}")
    print(f"  Power: {power:.8e}")


# =========================================================
# RECORD AUDIO
# =========================================================

print("Recording. Play pink noise / low-frequency noise near both mics.")

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

print_audio_levels("Mic A raw", mic_a)
print_audio_levels("Mic B raw", mic_b)


# =========================================================
# CHOOSE REFERENCE AND TARGET
# =========================================================

# For the side-by-side test, either choice is okay.
# Later, for the cardboard box setup:
# reference_signal = outside mic / closer to noise source
# target_signal    = inside mic / inside cardboard box

reference_signal = mic_a
target_signal = mic_b


# =========================================================
# FILTER TO ANC-RELEVANT LOW FREQUENCIES
# =========================================================

reference_low = bandpass(reference_signal, SAMPLE_RATE, low=LOWCUT, high=HIGHCUT)
target_low = bandpass(target_signal, SAMPLE_RATE, low=LOWCUT, high=HIGHCUT)


# =========================================================
# ALIGN SIGNALS USING YOUR CALIBRATED LAG
# =========================================================

reference_aligned, target_aligned = align_signals(
    reference_low,
    target_low,
    FIXED_LAG
)

aligned_corr = np.corrcoef(reference_aligned, target_aligned)[0, 1]

print("\nAfter filtering and alignment:")
print(f"Aligned correlation: {aligned_corr:.4f}")
print_audio_levels("Reference aligned", reference_aligned)
print_audio_levels("Target aligned", target_aligned)

reference_aligned = rms_normalise(reference_aligned)
target_aligned = rms_normalise(target_aligned)


# =========================================================
# RUN ADAPTIVE FILTER
# =========================================================
anc = off(filter_length=FILTER_LENGTH, learning_rate=LEARNING_RATE)

predicted_signal, error_signal = anc.process(
    reference_aligned,
    target_aligned)




# =========================================================
# PRINT METRICS
# =========================================================

def print_error_metrics(target_signal, predicted_signal, error_signal):
    target_signal = np.asarray(target_signal).ravel()
    predicted_signal = np.asarray(predicted_signal).ravel()
    error_signal = np.asarray(error_signal).ravel()

    baseline_power = np.mean(target_signal ** 2)
    predicted_power = np.mean(predicted_signal ** 2)
    error_power = np.mean(error_signal ** 2)

    normalised_error = error_power / baseline_power
    erle_db = 10 * np.log10(baseline_power / error_power)

    half = len(target_signal) // 2

    late_baseline_power = np.mean(target_signal[half:] ** 2)
    late_error_power = np.mean(error_signal[half:] ** 2)
    late_erle_db = 10 * np.log10(late_baseline_power / late_error_power)

    print("\n=== ADAPTIVE FILTER RESULTS ===")
    print(f"Baseline target power:       {baseline_power:.8e}")
    print(f"Predicted signal power:      {predicted_power:.8e}")
    print(f"Adaptive error power:        {error_power:.8e}")
    print(f"Normalised error:            {normalised_error:.4f}")
    print(f"ERLE:                        {erle_db:.2f} dB")
    print(f"Late-stage ERLE:             {late_erle_db:.2f} dB")
    print(f"Predicted-target corr:       {np.corrcoef(predicted_signal, target_signal)[0, 1]:.4f}")
    print(f"Error-target corr:           {np.corrcoef(error_signal, target_signal)[0, 1]:.4f}")
test_params(target_aligned, predicted_signal)
