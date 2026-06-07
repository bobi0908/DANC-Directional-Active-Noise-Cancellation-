import time
import numpy as np
import sounddevice as sd
from scipy.signal import butter, sosfilt, sosfiltfilt
from scipy.signal import correlate, correlation_lags


# =========================================================
# MODE
# =========================================================
# First run:
#     MODE = "calibrate"
#
# Then run:
#     MODE = "run"
# =========================================================

MODE = "run"   # change to "run" after calibration


# =========================================================
# AUDIO SETTINGS
# =========================================================

SAMPLE_RATE = 48000
DEVICE_ID = 0

INPUT_CHANNELS = 2
OUTPUT_CHANNELS = 2

ERROR_CHANNEL = 0      # inside box / cancellation point mic
REFERENCE_CHANNEL = 1   # outside/source-side mic

BLOCK_SIZE = 8192*2

# You found int16 output is the safe/working output type.
# Keep this low at first.
CALIBRATION_OUTPUT_LIMIT_INT16 = 50000
ANC_OUTPUT_LIMIT_INT16 = 10000

# Control command is internally clipped to this before conversion to int16.
MAX_COMMAND = 0.001


# =========================================================
# FILTER SETTINGS
# =========================================================

LOWCUT = 80
HIGHCUT = 500
USE_BANDPASS = False

SECONDARY_PATH_RUN_CENTER = 2104
SECONDARY_WINDOW_BEFORE = 10
SECONDARY_WINDOW_AFTER = 40

SECONDARY_PATH_LENGTH = 4096
CONTROL_FILTER_LENGTH = 16


# Start conservative for speaker-in-loop adaptation.
LEARNING_RATE = 0.0000001

EPSILON = 1e-6
LEAKAGE = 1e-6

# If ANC makes the error worse, stop and try changing this from -1.0 to +1.0.
UPDATE_SIGN = -1.0

SECONDARY_PATH_FILE = "secondary_path.npy"


# =========================================================
# SAFETY / TEST SETTINGS
# =========================================================

IGNORE_START_SECONDS = 1.0
BASELINE_SECONDS = 3.0

PRINT_INTERVAL_SECONDS = 0.25

# If error mic is below this, do not adapt.
# This avoids learning silence/electrical noise.
MIN_ERROR_RMS_FOR_ADAPTATION = 0.001


# =========================================================
# BASIC HELPERS
# =========================================================

def rms(x):
    x = np.asarray(x, dtype=np.float32).ravel()
    return float(np.sqrt(np.mean(x ** 2)))


def power(x):
    x = np.asarray(x, dtype=np.float32).ravel()
    return float(np.mean(x ** 2))


def make_bar(value, scale=50, width=25):
    filled = int(min(value * scale, 1.0) * width)
    return "#" * filled + "-" * (width - filled)


def bandpass_offline(x, sample_rate, low=80, high=1000, order=4):
    x = np.asarray(x, dtype=np.float32).ravel()
    nyquist = sample_rate / 2

    sos = butter(
        order,
        [low / nyquist, high / nyquist],
        btype="band",
        output="sos"
    )

    return sosfiltfilt(sos, x)


def estimate_lag(x, y, max_lag_seconds=0.2):
    """
    Estimate lag between command x and recorded mic signal y.

    Positive lag means y is delayed relative to x.
    """
    x = np.asarray(x, dtype=np.float32).ravel()
    y = np.asarray(y, dtype=np.float32).ravel()

    x = x - np.mean(x)
    y = y - np.mean(y)

    if np.std(x) > 0:
        x = x / np.std(x)
    if np.std(y) > 0:
        y = y / np.std(y)

    corr = correlate(y, x, mode="full")
    lags = correlation_lags(len(y), len(x), mode="full")

    max_lag = int(max_lag_seconds * SAMPLE_RATE)
    valid = np.abs(lags) <= max_lag

    corr = corr[valid]
    lags = lags[valid]

    best_i = np.argmax(np.abs(corr))
    best_lag = int(lags[best_i])
    best_corr = corr[best_i] / len(x)

    return best_lag, best_corr


class RealTimeBandpass:
    """
    Causal bandpass filter for live audio.

    This is used instead of sosfiltfilt because sosfiltfilt uses future samples,
    which is impossible in real time.
    """

    def __init__(self, sample_rate, low, high, channels, order=4):
        nyquist = sample_rate / 2

        self.sos = butter(
            order,
            [low / nyquist, high / nyquist],
            btype="band",
            output="sos"
        )

        self.channels = channels
        self.zi = np.zeros((channels, self.sos.shape[0], 2), dtype=np.float32)

    def process(self, block):
        block = np.asarray(block, dtype=np.float32)

        filtered = np.zeros_like(block)

        for ch in range(self.channels):
            filtered[:, ch], self.zi[ch] = sosfilt(
                self.sos,
                block[:, ch],
                zi=self.zi[ch]
            )

        return filtered


# =========================================================
# OFFLINE NLMS FOR SECONDARY PATH CALIBRATION
# =========================================================

class OfflineNLMSFilter:
    """
    Learns:
        input_signal -> target_signal

    Used here to learn:
        speaker command -> error mic signal
    """

    def __init__(self, filter_length=128, learning_rate=0.1, epsilon=1e-6):
        self.filter_length = filter_length
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        self.weights = np.zeros(filter_length, dtype=np.float32)
        self.history = np.zeros(filter_length, dtype=np.float32)

    def process_sample(self, input_sample, target_sample):
        self.history[1:] = self.history[:-1]
        self.history[0] = input_sample

        predicted = np.dot(self.weights, self.history)
        error = target_sample - predicted

        input_power = np.dot(self.history, self.history)

        self.weights += (
            self.learning_rate
            * error
            * self.history
            / (input_power + self.epsilon)
        )

        return predicted, error

    def process(self, input_signal, target_signal):
        input_signal = np.asarray(input_signal, dtype=np.float32).ravel()
        target_signal = np.asarray(target_signal, dtype=np.float32).ravel()

        length = min(len(input_signal), len(target_signal))

        predicted = np.zeros(length, dtype=np.float32)
        error = np.zeros(length, dtype=np.float32)

        for n in range(length):
            predicted[n], error[n] = self.process_sample(
                input_signal[n],
                target_signal[n]
            )

        return predicted, error


# =========================================================
# ONLINE FxLMS FILTER
# =========================================================

class OnlineFxLMSFilter:
    """
    Faster speaker-in-the-loop FxLMS filter.

    This version keeps the secondary-path delay using a circular delay buffer,
    instead of shifting a huge array every sample.
    """

    def __init__(
        self,
        control_length,
        secondary_path,
        secondary_start=0,
        learning_rate=0.002,
        epsilon=1e-6,
        leakage=1e-6,
        update_sign=-1.0,
        max_command=0.5
    ):
        self.control_length = int(control_length)
        self.secondary_path = np.asarray(secondary_path, dtype=np.float32).ravel()
        self.secondary_length = len(self.secondary_path)
        self.secondary_start = int(secondary_start)

        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.leakage = leakage
        self.update_sign = update_sign
        self.max_command = max_command

        self.control_weights = np.zeros(self.control_length, dtype=np.float32)

        # History for generating speaker command
        self.reference_history = np.zeros(self.control_length, dtype=np.float32)

        # Circular buffer to implement the large delay before the useful secondary-path window
        if self.secondary_start > 0:
            self.delay_buffer = np.zeros(self.secondary_start, dtype=np.float32)
        else:
            self.delay_buffer = np.zeros(1, dtype=np.float32)

        self.delay_index = 0

        # Short history only for the cropped secondary-path window
        self.secondary_history = np.zeros(self.secondary_length, dtype=np.float32)

        # History used for FxLMS update
        self.filtered_reference_history = np.zeros(self.control_length, dtype=np.float32)

    def reset(self):
        self.control_weights[:] = 0
        self.reference_history[:] = 0
        self.delay_buffer[:] = 0
        self.delay_index = 0
        self.secondary_history[:] = 0
        self.filtered_reference_history[:] = 0

    def process_sample(self, reference_sample, error_sample, adapt=True):
        # Main reference history for output generation
        self.reference_history[1:] = self.reference_history[:-1]
        self.reference_history[0] = reference_sample

        # Speaker command
        command = np.dot(self.control_weights, self.reference_history)
        command = np.clip(command, -self.max_command, self.max_command)

        # Apply the large delay efficiently using a circular buffer
        if self.secondary_start > 0:
            delayed_reference = self.delay_buffer[self.delay_index]
            self.delay_buffer[self.delay_index] = reference_sample
            self.delay_index = (self.delay_index + 1) % self.secondary_start
        else:
            delayed_reference = reference_sample

        # Now filter the delayed reference through the short secondary-path window
        self.secondary_history[1:] = self.secondary_history[:-1]
        self.secondary_history[0] = delayed_reference

        filtered_reference_sample = np.dot(
            self.secondary_path,
            self.secondary_history
        )

        self.filtered_reference_history[1:] = self.filtered_reference_history[:-1]
        self.filtered_reference_history[0] = filtered_reference_sample

        if adapt:
            filtered_power = np.dot(
                self.filtered_reference_history,
                self.filtered_reference_history
            )

            update = (
                self.learning_rate
                * error_sample
                * self.filtered_reference_history
                / (filtered_power + self.epsilon)
            )

            self.control_weights = (
                (1.0 - self.leakage) * self.control_weights
                + self.update_sign * update
            )

        return command

    def process_block(self, reference_block, error_block, adapt=True):
        reference_block = np.asarray(reference_block, dtype=np.float32).ravel()
        error_block = np.asarray(error_block, dtype=np.float32).ravel()

        length = min(len(reference_block), len(error_block))
        command_block = np.zeros(length, dtype=np.float32)

        for n in range(length):
            command_block[n] = self.process_sample(
                reference_block[n],
                error_block[n],
                adapt=adapt
            )

        return command_block


# =========================================================
# MODE 1: CALIBRATE SECONDARY PATH
# =========================================================

def calibrate_secondary_path():
    """
    Learns the path:

        speaker command -> error mic

    Do this with the speaker and error mic in their final positions.
    Keep external noise quiet during this calibration.
    """

    duration = 12
    n_samples = int(duration * SAMPLE_RATE)

    print("Secondary path calibration.")
    print("Keep the room quiet except for the speaker calibration noise.")
    print("Speaker and error mic should already be in final positions.")
    print("Recording/playing calibration signal...")

    rng = np.random.default_rng(123)

    # Normalised internal speaker command, not int16 yet.
    command = rng.normal(0, 1, n_samples).astype(np.float32)

    if USE_BANDPASS:
        command = bandpass_offline(command, SAMPLE_RATE, LOWCUT, HIGHCUT)

    # Normalise peak then keep it modest.
    peak = np.max(np.abs(command))
    if peak > 0:
        command = command / peak

    command = 0.4 * command

    # Convert command to int16 speaker output.
    speaker_int16 = (command * CALIBRATION_OUTPUT_LIMIT_INT16).astype(np.int16)

    # Send same output to both channels because the device exposes 2 outputs.
    output = np.column_stack([speaker_int16, speaker_int16]).astype(np.int16)

    recording = sd.playrec(
        output,
        samplerate=SAMPLE_RATE,
        channels=INPUT_CHANNELS,
        device=DEVICE_ID,
        dtype="float32",
        blocking=True
    )

    # Remove startup transient
    start = int(IGNORE_START_SECONDS * SAMPLE_RATE)

    command = command[start:]
    error_mic = recording[start:, ERROR_CHANNEL]

    lag, lag_corr = estimate_lag(command, error_mic)

    print("\nSpeaker-to-error-mic lag estimate:")
    print("Lag:", lag, "samples")
    print("Lag seconds:", lag / SAMPLE_RATE)
    print("Lag correlation:", lag_corr)

    if abs(lag) > SECONDARY_PATH_LENGTH:
        print("WARNING: lag is bigger than SECONDARY_PATH_LENGTH.")
        print("Increase SECONDARY_PATH_LENGTH.")

    if USE_BANDPASS:
        error_mic = bandpass_offline(error_mic, SAMPLE_RATE, LOWCUT, HIGHCUT)

    print("\nCalibration recording levels:")
    print("Command RMS:", rms(command))
    print("Error mic RMS:", rms(error_mic))
    print("Error mic peak:", np.max(np.abs(error_mic)))

    nlms = OfflineNLMSFilter(
        filter_length=SECONDARY_PATH_LENGTH,
        learning_rate=0.1
    )

    predicted, error = nlms.process(command, error_mic)

    secondary_path = nlms.weights.copy()

    np.save(SECONDARY_PATH_FILE, secondary_path)

    baseline_power = power(error_mic)
    residual_power = power(error)

    erle = 10 * np.log10(baseline_power / residual_power)

    corr = np.corrcoef(predicted, error_mic)[0, 1]

    print("\nSecondary path learned and saved to:", SECONDARY_PATH_FILE)
    print(f"Secondary path fit ERLE: {erle:.2f} dB")
    print(f"Predicted/error-mic correlation: {corr:.4f}")
    print("Secondary path first 10 taps:")
    print(secondary_path[:10])


# =========================================================
# MODE 2: RUN LIVE SPEAKER ANC
# =========================================================

def run_live_anc():
    full_secondary_path = np.load(SECONDARY_PATH_FILE)

    secondary_peak_index = int(np.argmax(np.abs(full_secondary_path)))

    secondary_start = max(
        0,
        secondary_peak_index - SECONDARY_WINDOW_BEFORE
    )

    secondary_end = min(
        len(full_secondary_path),
        secondary_peak_index + SECONDARY_WINDOW_AFTER
    )

    secondary_path = full_secondary_path[secondary_start:secondary_end]

    print("Loaded secondary path:", SECONDARY_PATH_FILE)
    print("Full secondary path length:", len(full_secondary_path))
    print("Strongest secondary path tap:", secondary_peak_index)
    print("Using delay-aware secondary path window:")
    print("  start:", secondary_start)
    print("  end:", secondary_end)
    print("  length:", len(secondary_path))

    anc = OnlineFxLMSFilter(
        control_length=CONTROL_FILTER_LENGTH,
        secondary_path=secondary_path,
        secondary_start=secondary_start,
        learning_rate=LEARNING_RATE,
        epsilon=EPSILON,
        leakage=LEAKAGE,
        update_sign=UPDATE_SIGN,
        max_command=MAX_COMMAND
    )

    if USE_BANDPASS:
        bandpass_filter = RealTimeBandpass(
            sample_rate=SAMPLE_RATE,
            low=LOWCUT,
            high=HIGHCUT,
            channels=INPUT_CHANNELS
        )
    else:
        bandpass_filter = None

    ignore_blocks = int(IGNORE_START_SECONDS * SAMPLE_RATE / BLOCK_SIZE)
    baseline_blocks = int(BASELINE_SECONDS * SAMPLE_RATE / BLOCK_SIZE)

    state = {
        "block_count": 0,
        "baseline_power_sum": 0.0,
        "baseline_count": 0,
        "baseline_power": None,
        "latest_erle": np.nan,
        "latest_ref_rms": 0.0,
        "latest_err_rms": 0.0,
        "latest_out_max": 0,
        "latest_adapt": False,
        "latest_status": "",
        "baseline_just_finished": False,
        "status_count": 0,
    }

    def callback(indata, outdata, frames, time_info, status):
        if status:
            state["status_count"] += 1
            state["latest_status"] = str(status)
        else:
            state["latest_status"] = ""

        state["block_count"] += 1

        # Default output is silence.
        outdata[:] = 0

        block = np.asarray(indata, dtype=np.float32)

        # Ignore startup transient.
        if state["block_count"] <= ignore_blocks:
            return

        if USE_BANDPASS:
            block = bandpass_filter.process(block)

        reference_block = block[:, REFERENCE_CHANNEL]
        error_block = block[:, ERROR_CHANNEL]

        error_rms = rms(error_block)
        reference_rms = rms(reference_block)

        # Baseline period: output silence, measure uncancelled error.
        if state["block_count"] <= ignore_blocks + baseline_blocks:
            state["baseline_power_sum"] += power(error_block)
            state["baseline_count"] += 1
            state["latest_ref_rms"] = reference_rms
            state["latest_err_rms"] = error_rms
            state["latest_adapt"] = False
            return

        if state["baseline_power"] is None:
            state["baseline_power"] = (
                state["baseline_power_sum"]
                / max(state["baseline_count"], 1)
            )
            anc.reset()
            state["baseline_just_finished"] = True

        adapt = error_rms >= MIN_ERROR_RMS_FOR_ADAPTATION

        command_block = anc.process_block(
            reference_block,
            error_block,
            adapt=adapt
        )

        # Convert normalised command to int16 output.
        command_block = np.clip(command_block, -MAX_COMMAND, MAX_COMMAND)

        output_int16 = (
            command_block * ANC_OUTPUT_LIMIT_INT16
        ).astype(np.int16)

        # Send same signal to both output channels.
        outdata[:, 0] = output_int16
        outdata[:, 1] = output_int16

        error_power_now = power(error_block)

        # Safety shutoff: if ANC makes the error much worse, mute output.
        if state["baseline_power"] is not None:
            if error_power_now > 4.0 * state["baseline_power"]:
                outdata[:] = 0
                state["latest_status"] = "SAFETY MUTE: error too high"
                state["latest_adapt"] = False
                return

        if state["baseline_power"] > 0 and error_power_now > 0:
            erle = 10 * np.log10(state["baseline_power"] / error_power_now)
        else:
            erle = np.nan

        state["latest_erle"] = erle
        state["latest_ref_rms"] = reference_rms
        state["latest_err_rms"] = error_rms
        state["latest_out_max"] = int(np.max(np.abs(output_int16)))
        state["latest_adapt"] = adapt

    print("\nStarting live speaker ANC.")
    print("First seconds: output silence for baseline.")
    print("Then ANC starts.")
    print("Press Ctrl+C to stop immediately if it gets loud/unstable.\n")

    print("Settings:")
    print("  ANC_OUTPUT_LIMIT_INT16 =", ANC_OUTPUT_LIMIT_INT16)
    print("  MAX_COMMAND =", MAX_COMMAND)
    print("  LEARNING_RATE =", LEARNING_RATE)
    print("  UPDATE_SIGN =", UPDATE_SIGN)
    print("  CONTROL_FILTER_LENGTH =", CONTROL_FILTER_LENGTH)
    print("  SECONDARY_PATH_LENGTH =", SECONDARY_PATH_LENGTH)

    try:
        with sd.Stream(
            device=DEVICE_ID,
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=(INPUT_CHANNELS, OUTPUT_CHANNELS),
            dtype=("float32", "int16"),
            callback=callback,
            latency="high"
        ):
            last_print = 0

            while True:
                now = time.time()

                if state["baseline_just_finished"]:
                    print("\nBaseline measured. ANC starting...\n")
                    state["baseline_just_finished"] = False

                if now - last_print >= PRINT_INTERVAL_SECONDS:
                    last_print = now

                    ref_bar = make_bar(state["latest_ref_rms"])
                    err_bar = make_bar(state["latest_err_rms"])

                    adapt_text = "ADAPT" if state["latest_adapt"] else "NOADAPT"

                    print(
                        f"\r"
                        f"ERLE={state['latest_erle']:6.2f} dB | "
                        f"REF [{ref_bar}] {state['latest_ref_rms']:.5f} | "
                        f"ERR [{err_bar}] {state['latest_err_rms']:.5f} | "
                        f"OUT={state['latest_out_max']:4d} | "
                        f"{adapt_text} | "
                        f"{state['latest_status'][:30]}",
                        f"STAT={state['status_count']} | ",
                        end="",
                        flush=True
                    )

                time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nStopped live ANC.")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    print("Using device:")
    print(sd.query_devices(DEVICE_ID))

    if MODE == "calibrate":
        calibrate_secondary_path()

    elif MODE == "run":
        run_live_anc()

    else:
        raise ValueError("MODE must be either 'calibrate' or 'run'.")
