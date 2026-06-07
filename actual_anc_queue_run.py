import time
from collections import deque

import numpy as np
import sounddevice as sd


# =========================================================
# AUDIO SETTINGS
# =========================================================

SAMPLE_RATE = 48000
DEVICE_ID = 0

INPUT_CHANNELS = 2
OUTPUT_CHANNELS = 2

# Your corrected order
ERROR_CHANNEL = 0       # inside box / cancellation point mic
REFERENCE_CHANNEL = 1   # outside/source-side mic

BLOCK_SIZE = 4096       # start here; if stable, later try 2048

SECONDARY_PATH_FILE = "secondary_path.npy"


# =========================================================
# RUN SETTINGS
# =========================================================

ANC_OUTPUT_LIMIT_INT16 = 10000
MAX_COMMAND = 0.003

CONTROL_FILTER_LENGTH = 16

SECONDARY_WINDOW_BEFORE = 10
SECONDARY_WINDOW_AFTER = 40

LEARNING_RATE = 0.000001
EPSILON = 1e-6
LEAKAGE = 1e-6
UPDATE_SIGN = -1.0

IGNORE_START_SECONDS = 1.0
BASELINE_SECONDS = 8.0
PRINT_INTERVAL_SECONDS = 0.25

MIN_ERROR_RMS_FOR_ADAPTATION = 0.0003

# Safety: mute if current error power is much worse than baseline
ENABLE_SAFETY_MUTE = False
SAFETY_FACTOR = 25.0


# =========================================================
# QUEUES BETWEEN CALLBACK AND MAIN LOOP
# =========================================================

input_blocks = deque(maxlen=8)
output_blocks = deque(maxlen=4)

zero_output_block = np.zeros((BLOCK_SIZE, OUTPUT_CHANNELS), dtype=np.int16)

state = {
    "callback_blocks": 0,
    "portaudio_status_count": 0,
    "software_output_misses": 0,
    "last_portaudio_status": "",

    "processed_blocks": 0,
    "baseline_power_sum": 0.0,
    "baseline_count": 0,
    "baseline_power": None,

    "latest_erle": np.nan,
    "latest_ref_rms": 0.0,
    "latest_err_rms": 0.0,
    "latest_out_max": 0,
    "latest_adapt": False,
    "latest_status": "starting",
}


# =========================================================
# HELPERS
# =========================================================

def rms(x):
    x = np.asarray(x, dtype=np.float32).ravel()
    return float(np.sqrt(np.mean(x ** 2)))


def power(x):
    x = np.asarray(x, dtype=np.float32).ravel()
    return float(np.mean(x ** 2))


def make_bar(value, scale=300, width=25):
    filled = int(min(value * scale, 1.0) * width)
    return "#" * filled + "-" * (width - filled)


# =========================================================
# FASTER FxLMS FILTER
# =========================================================

class OnlineFxLMSFilter:
    """
    FxLMS with a delay-aware cropped secondary path.

    The large secondary-path delay is handled with a circular delay buffer,
    not by shifting a huge array every sample.
    """

    def __init__(
        self,
        control_length,
        secondary_path,
        secondary_start=0,
        learning_rate=0.000001,
        epsilon=1e-6,
        leakage=1e-6,
        update_sign=-1.0,
        max_command=0.003,
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

        self.reference_history = np.zeros(self.control_length, dtype=np.float32)

        if self.secondary_start > 0:
            self.delay_buffer = np.zeros(self.secondary_start, dtype=np.float32)
        else:
            self.delay_buffer = np.zeros(1, dtype=np.float32)

        self.delay_index = 0

        self.secondary_history = np.zeros(self.secondary_length, dtype=np.float32)
        self.filtered_reference_history = np.zeros(self.control_length, dtype=np.float32)

    def reset(self):
        self.control_weights[:] = 0
        self.reference_history[:] = 0
        self.delay_buffer[:] = 0
        self.delay_index = 0
        self.secondary_history[:] = 0
        self.filtered_reference_history[:] = 0

    def process_sample(self, reference_sample, error_sample, adapt=True):
        # Reference history for generating speaker command
        self.reference_history[1:] = self.reference_history[:-1]
        self.reference_history[0] = reference_sample

        command = np.dot(self.control_weights, self.reference_history)
        command = np.clip(command, -self.max_command, self.max_command)

        # Delay-aware secondary path filtering
        if self.secondary_start > 0:
            delayed_reference = self.delay_buffer[self.delay_index]
            self.delay_buffer[self.delay_index] = reference_sample
            self.delay_index = (self.delay_index + 1) % self.secondary_start
        else:
            delayed_reference = reference_sample

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
                adapt=adapt,
            )

        return command_block


# =========================================================
# AUDIO CALLBACK: KEEP THIS TINY
# =========================================================

def audio_callback(indata, outdata, frames, time_info, status):
    state["callback_blocks"] += 1

    if status:
        state["portaudio_status_count"] += 1
        state["last_portaudio_status"] = str(status)
    else:
        state["last_portaudio_status"] = ""

    # Play the next already-computed output block.
    # If none is ready, output silence.
    try:
        next_output = output_blocks.popleft()
        outdata[:] = next_output
    except IndexError:
        outdata[:] = 0
        state["software_output_misses"] += 1

    # Store input for the main loop to process.
    # Copy is necessary because sounddevice reuses the input buffer.
    input_blocks.append(indata.copy())


# =========================================================
# MAIN PROCESSING LOOP
# =========================================================

def main():
    print("Using device:")
    print(sd.query_devices(DEVICE_ID))

    full_secondary_path = np.load(SECONDARY_PATH_FILE)

    secondary_peak_index = int(np.argmax(np.abs(full_secondary_path)))

    secondary_start = max(0, secondary_peak_index - SECONDARY_WINDOW_BEFORE)
    secondary_end = min(
        len(full_secondary_path),
        secondary_peak_index + SECONDARY_WINDOW_AFTER,
    )

    secondary_path = full_secondary_path[secondary_start:secondary_end]

    print("\nLoaded secondary path:", SECONDARY_PATH_FILE)
    print("Full secondary path length:", len(full_secondary_path))
    print("Strongest secondary path tap:", secondary_peak_index)
    print("Using secondary path window:")
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
        max_command=MAX_COMMAND,
    )

    ignore_blocks = int(np.ceil(IGNORE_START_SECONDS * SAMPLE_RATE / BLOCK_SIZE))
    baseline_blocks = int(np.ceil(BASELINE_SECONDS * SAMPLE_RATE / BLOCK_SIZE))

    print("\nStarting queued ANC run.")
    print("Callback only handles audio I/O.")
    print("Main loop does FxLMS processing.")
    print("Press Ctrl+C to stop.\n")

    print("Settings:")
    print("  BLOCK_SIZE =", BLOCK_SIZE)
    print("  CONTROL_FILTER_LENGTH =", CONTROL_FILTER_LENGTH)
    print("  ANC_OUTPUT_LIMIT_INT16 =", ANC_OUTPUT_LIMIT_INT16)
    print("  MAX_COMMAND =", MAX_COMMAND)
    print("  LEARNING_RATE =", LEARNING_RATE)
    print("  UPDATE_SIGN =", UPDATE_SIGN)

    last_print = 0

    try:
        with sd.Stream(
            device=DEVICE_ID,
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=(INPUT_CHANNELS, OUTPUT_CHANNELS),
            dtype=("float32", "int16"),
            callback=audio_callback,
            latency="high",
        ):
            while True:
                try:
                    block = input_blocks.popleft()
                except IndexError:
                    time.sleep(0.001)
                    continue

                state["processed_blocks"] += 1

                reference_block = block[:, REFERENCE_CHANNEL]
                error_block = block[:, ERROR_CHANNEL]

                reference_rms = rms(reference_block)
                error_rms = rms(error_block)

                # Startup ignore period
                if state["processed_blocks"] <= ignore_blocks:
                    output_blocks.append(zero_output_block.copy())
                    state["latest_status"] = "ignoring startup"
                    continue

                # Baseline period: speaker silent
                if state["processed_blocks"] <= ignore_blocks + baseline_blocks:
                    state["baseline_power_sum"] += power(error_block)
                    state["baseline_count"] += 1

                    output_blocks.append(zero_output_block.copy())

                    state["latest_ref_rms"] = reference_rms
                    state["latest_err_rms"] = error_rms
                    state["latest_out_max"] = 0
                    state["latest_adapt"] = False
                    state["latest_status"] = "baseline"
                    continue

                if state["baseline_power"] is None:
                    state["baseline_power"] = (
                        state["baseline_power_sum"]
                        / max(state["baseline_count"], 1)
                    )
                    anc.reset()
                    state["latest_status"] = "ANC started"

                adapt = error_rms >= MIN_ERROR_RMS_FOR_ADAPTATION

                command_block = anc.process_block(
                    reference_block,
                    error_block,
                    adapt=adapt,
                )

                command_block = np.clip(command_block, -MAX_COMMAND, MAX_COMMAND)

                output_mono = (
                    command_block * ANC_OUTPUT_LIMIT_INT16
                ).astype(np.int16)

                output_block = np.column_stack([output_mono, output_mono]).astype(np.int16)

                error_power_now = power(error_block)

                # Safety mute
                if (
                    state["baseline_power"] is not None
                    and error_power_now > SAFETY_FACTOR * state["baseline_power"]
                ):
                    output_block = zero_output_block.copy()
                    state["latest_status"] = "SAFETY MUTE"
                    adapt = False

                output_blocks.append(output_block)

                if state["baseline_power"] > 0 and error_power_now > 0:
                    erle = 10 * np.log10(state["baseline_power"] / error_power_now)
                else:
                    erle = np.nan

                state["latest_erle"] = erle
                state["latest_ref_rms"] = reference_rms
                state["latest_err_rms"] = error_rms
                state["latest_out_max"] = int(np.max(np.abs(output_block)))
                state["latest_adapt"] = adapt

                now = time.time()

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
                        f"PA_STAT={state['portaudio_status_count']} | "
                        f"SW_MISS={state['software_output_misses']} | "
                        f"{state['latest_status']} | "
                        f"{state['last_portaudio_status'][:25]}",
                        end="",
                        flush=True,
                    )

    except KeyboardInterrupt:
        print("\nStopped queued ANC run.")


if __name__ == "__main__":
    main()
