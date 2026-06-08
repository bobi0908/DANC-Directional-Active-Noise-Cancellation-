import csv
import ctypes
import os
import sys
import threading
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
CONTROL_WEIGHTS_FILE = "control_weights.npy"


# =========================================================
# RUN SETTINGS
# =========================================================

ANC_OUTPUT_LIMIT_INT16 = 10000
MAX_COMMAND = 0.01

CONTROL_FILTER_LENGTH = 16

SECONDARY_WINDOW_BEFORE = 10
SECONDARY_WINDOW_AFTER = 40

LEARNING_RATE = 1e-5
EPSILON = 1e-6
LEAKAGE = 1e-6
UPDATE_SIGN = -1.0

IGNORE_START_SECONDS = 1.0
BASELINE_SECONDS = 8.0
PRINT_INTERVAL_SECONDS = 0.25

MIN_ERROR_RMS_FOR_ADAPTATION = 0.0003

# Safety: mute if current error power is much worse than baseline
ENABLE_SAFETY_MUTE = True
SAFETY_FACTOR = 25.0

# Live demo on/off toggle: press ENTER in this terminal to switch ANC on and
# off while it runs, so an audience can watch the error-mic level jump up
# (system off) and drop again (system on). Purely a presentation control --
# "off" just outputs silence and freezes the filter, reusing the same path as
# the safety mute. Defaults to ON, and if the key-listener ever stops the flag
# simply stays at its last value, so the system keeps cancelling no matter what.
ENABLE_ANC_TOGGLE = True
START_WITH_ANC_ON = True

# Shared file the live graph window uses to request ANC on/off, so you can
# toggle from the graph (native window or remote browser) as well as with the
# ENTER key here. This process treats the file as the single source of truth.
ANC_CONTROL_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "anc_control.txt"
)


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
    "baseline_rms": None,

    "latest_erle": np.nan,
    "latest_ref_rms": 0.0,
    "latest_err_rms": 0.0,
    "latest_out_max": 0,
    "latest_adapt": False,
    "latest_status": "starting",

    # Live demo on/off toggle (see ENABLE_ANC_TOGGLE). Always starts True so
    # that, with the feature disabled or the listener dead, ANC stays on.
    "anc_enabled": True,
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


def write_anc_control(enabled):
    """Atomically record the requested ANC on/off state to ANC_CONTROL_FILE.

    Written via a temp file + rename so a reader never sees a half-written
    value. Never raises -- if it fails, the toggle just doesn't propagate.
    """
    try:
        tmp = ANC_CONTROL_FILE + ".tmp"
        with open(tmp, "w") as f:
            f.write("on" if enabled else "off")
        os.replace(tmp, ANC_CONTROL_FILE)
    except Exception:
        pass


def read_anc_control(default):
    """Return the requested state from ANC_CONTROL_FILE, or `default` if the
    file is missing or unreadable. Never raises, so an I/O hiccup just keeps
    the current state -- the system never flips off by accident.
    """
    try:
        with open(ANC_CONTROL_FILE) as f:
            value = f.read().strip().lower()
        if value == "on":
            return True
        if value == "off":
            return False
    except Exception:
        pass
    return default


def anc_toggle_listener():
    """Background thread: each press of ENTER flips ANC on/off for the demo.

    Runs as a daemon so it dies with the process. It flips the requested state
    in the shared control file (the main loop applies it), so the ENTER key and
    the graph's toggle button stay in agreement. If stdin closes or anything
    goes wrong it simply stops listening.
    """
    try:
        for _line in sys.stdin:
            new_state = not read_anc_control(state["anc_enabled"])
            write_anc_control(new_state)
            if new_state:
                print("\n>>> ANC switched ON  -- cancelling <<<")
            else:
                print("\n>>> ANC switched OFF -- error mic now hears the "
                      "full, un-cancelled noise <<<")
    except Exception:
        pass


# =========================================================
# FASTER FxLMS FILTER (per-sample core runs in libfxlms.so)
# =========================================================
#
# process_sample() used to run ~48,000 times/sec in pure Python/numpy
# (BLOCK_SIZE=4096 samples * ~11.7 blocks/sec), each call paying numpy
# dispatch overhead on tiny 16-50 element arrays. That overhead -- not
# the actual FLOPs -- is what made the callback miss its deadline.
#
# The per-sample math now runs in fxlms.c (compiled to libfxlms.so via
# the Makefile) as a tight loop over raw float buffers. This class keeps
# the same interface as before; numpy still owns all the persistent
# buffers, it just hands raw pointers to C for the hot loop.

_FXLMS_LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libfxlms.so")
_fxlms_lib = ctypes.CDLL(_FXLMS_LIB_PATH)

_c_float_p = ctypes.POINTER(ctypes.c_float)


class _FxLMSState(ctypes.Structure):
    _fields_ = [
        ("control_length", ctypes.c_int),
        ("secondary_length", ctypes.c_int),
        ("secondary_start", ctypes.c_int),
        ("delay_length", ctypes.c_int),
        ("delay_index", ctypes.c_int),

        ("learning_rate", ctypes.c_float),
        ("epsilon", ctypes.c_float),
        ("leakage", ctypes.c_float),
        ("update_sign", ctypes.c_float),
        ("max_command", ctypes.c_float),

        ("control_weights", _c_float_p),
        ("reference_history", _c_float_p),
        ("filtered_reference_history", _c_float_p),
        ("secondary_path", _c_float_p),
        ("secondary_history", _c_float_p),
        ("delay_buffer", _c_float_p),
    ]


_fxlms_lib.fxlms_reset.argtypes = [ctypes.POINTER(_FxLMSState)]
_fxlms_lib.fxlms_reset.restype = None

_fxlms_lib.fxlms_process_block.argtypes = [
    ctypes.POINTER(_FxLMSState),
    _c_float_p,
    _c_float_p,
    _c_float_p,
    ctypes.c_int,
    ctypes.c_int,
]
_fxlms_lib.fxlms_process_block.restype = None


def _float_ptr(array):
    return array.ctypes.data_as(_c_float_p)


class OnlineFxLMSFilter:
    """
    FxLMS with a delay-aware cropped secondary path.

    The large secondary-path delay is handled with a circular delay buffer,
    not by shifting a huge array every sample. The per-sample update itself
    runs in C (libfxlms.so); this class owns the numpy buffers that back it.
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
        self.secondary_path = np.asarray(secondary_path, dtype=np.float32).ravel().copy()
        self.secondary_length = len(self.secondary_path)
        self.secondary_start = int(secondary_start)
        delay_length = self.secondary_start if self.secondary_start > 0 else 1

        self.control_weights = np.zeros(self.control_length, dtype=np.float32)
        self.reference_history = np.zeros(self.control_length, dtype=np.float32)
        self.filtered_reference_history = np.zeros(self.control_length, dtype=np.float32)
        self.secondary_history = np.zeros(self.secondary_length, dtype=np.float32)
        self.delay_buffer = np.zeros(delay_length, dtype=np.float32)

        # Buffers below are owned by numpy; the struct just borrows raw
        # pointers into them. They must outlive self._state, which they do
        # since they're all attributes of the same object.
        self._state = _FxLMSState(
            control_length=self.control_length,
            secondary_length=self.secondary_length,
            secondary_start=self.secondary_start,
            delay_length=delay_length,
            delay_index=0,

            learning_rate=float(learning_rate),
            epsilon=float(epsilon),
            leakage=float(leakage),
            update_sign=float(update_sign),
            max_command=float(max_command),

            control_weights=_float_ptr(self.control_weights),
            reference_history=_float_ptr(self.reference_history),
            filtered_reference_history=_float_ptr(self.filtered_reference_history),
            secondary_path=_float_ptr(self.secondary_path),
            secondary_history=_float_ptr(self.secondary_history),
            delay_buffer=_float_ptr(self.delay_buffer),
        )

    def reset(self):
        _fxlms_lib.fxlms_reset(ctypes.byref(self._state))

    def process_block(self, reference_block, error_block, adapt=True):
        reference_block = np.ascontiguousarray(reference_block, dtype=np.float32)
        error_block = np.ascontiguousarray(error_block, dtype=np.float32)

        length = min(len(reference_block), len(error_block))
        command_block = np.empty(length, dtype=np.float32)

        _fxlms_lib.fxlms_process_block(
            ctypes.byref(self._state),
            _float_ptr(reference_block),
            _float_ptr(error_block),
            _float_ptr(command_block),
            length,
            1 if adapt else 0,
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

    saved_weights = None
    if os.path.exists(CONTROL_WEIGHTS_FILE):
        loaded_weights = np.load(CONTROL_WEIGHTS_FILE)
        if len(loaded_weights) == anc.control_length:
            saved_weights = loaded_weights.astype(np.float32)
            print(f"\nFound {CONTROL_WEIGHTS_FILE} -- will resume from its saved control weights.")
        else:
            print(
                f"\nIgnoring {CONTROL_WEIGHTS_FILE}: saved length {len(loaded_weights)} "
                f"!= CONTROL_FILTER_LENGTH {anc.control_length}. Starting from zero."
            )

    ignore_blocks = int(np.ceil(IGNORE_START_SECONDS * SAMPLE_RATE / BLOCK_SIZE))
    baseline_blocks = int(np.ceil(BASELINE_SECONDS * SAMPLE_RATE / BLOCK_SIZE))

    print("\nStarting queued ANC run.")
    print("Callback only handles audio I/O.")
    print("Main loop does FxLMS processing.")
    print("Press Ctrl+C to stop.")
    if ENABLE_ANC_TOGGLE:
        print("Press ENTER to switch ANC ON/OFF live "
              "(wait until it says 'ANC running' first).")
    print()

    print("Settings:")
    print("  BLOCK_SIZE =", BLOCK_SIZE)
    print("  CONTROL_FILTER_LENGTH =", CONTROL_FILTER_LENGTH)
    print("  ANC_OUTPUT_LIMIT_INT16 =", ANC_OUTPUT_LIMIT_INT16)
    print("  MAX_COMMAND =", MAX_COMMAND)
    print("  LEARNING_RATE =", LEARNING_RATE)
    print("  UPDATE_SIGN =", UPDATE_SIGN)

    last_print = 0

    log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"anc_log_{int(time.time())}.csv",
    )
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "elapsed_seconds", "erle_db", "baseline_rms", "ref_rms", "err_rms",
        "out_max", "adapt", "status",
    ])
    print(f"Logging readings to: {log_path}\n")

    state["anc_enabled"] = True
    if ENABLE_ANC_TOGGLE:
        state["anc_enabled"] = START_WITH_ANC_ON
        write_anc_control(state["anc_enabled"])  # seed the shared control file
        threading.Thread(target=anc_toggle_listener, daemon=True).start()

    run_start = time.time()

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

                # Apply the latest on/off request from the shared control file
                # (written by the ENTER key or the graph's toggle button). The
                # file is the single source of truth; a missing/garbled file
                # leaves the current state untouched.
                if ENABLE_ANC_TOGGLE:
                    state["anc_enabled"] = read_anc_control(state["anc_enabled"])

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
                    state["baseline_rms"] = float(np.sqrt(state["baseline_power"]))
                    anc.reset()
                    if saved_weights is not None:
                        anc.control_weights[:] = saved_weights
                    state["latest_status"] = "ANC started"

                # Freeze adaptation while ANC is manually switched off, so the
                # filter resumes exactly where it left off when switched back on.
                adapt = (
                    state["anc_enabled"]
                    and error_rms >= MIN_ERROR_RMS_FOR_ADAPTATION
                )

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

                # Manual demo toggle takes precedence: when switched off we
                # output silence (reusing the safety-mute path), so the error
                # mic hears the full noise and the audience can watch the level
                # jump back up.
                if not state["anc_enabled"]:
                    output_block = zero_output_block.copy()
                    state["latest_status"] = "ANC OFF"
                    adapt = False
                # Safety mute
                elif (
                    ENABLE_SAFETY_MUTE
                    and state["baseline_power"] is not None
                    and error_power_now > SAFETY_FACTOR * state["baseline_power"]
                ):
                    output_block = zero_output_block.copy()
                    state["latest_status"] = "SAFETY MUTE"
                    adapt = False
                else:
                    state["latest_status"] = "ANC running"

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

                log_writer.writerow([
                    f"{time.time() - run_start:.3f}",
                    f"{erle:.4f}" if not np.isnan(erle) else "",
                    f"{state['baseline_rms']:.6f}" if state["baseline_rms"] is not None else "",
                    f"{reference_rms:.6f}",
                    f"{error_rms:.6f}",
                    state["latest_out_max"],
                    int(adapt),
                    state["latest_status"],
                ])

                now = time.time()

                if now - last_print >= PRINT_INTERVAL_SECONDS:
                    last_print = now
                    log_file.flush()

                    adapt_text = "ADAPT" if state["latest_adapt"] else "NOADAPT"

                    # Kept short and padded to a fixed width on purpose: a line
                    # longer than the terminal wraps onto a second row, and "\r"
                    # then only rewinds to the start of that wrapped row -- not
                    # the start of the logical line -- which breaks the
                    # single-line-refresh effect and prints a wall of text.
                    line = (
                        f"ERLE={state['latest_erle']:6.2f}dB | "
                        f"REF={state['latest_ref_rms']:.4f} | "
                        f"ERR={state['latest_err_rms']:.4f} | "
                        f"OUT={state['latest_out_max']:3d} | "
                        f"{adapt_text:7s} | "
                        f"{state['latest_status'][:16]}"
                    )
                    print(f"\r{line:<76}", end="", flush=True)

    except KeyboardInterrupt:
        print("\nStopped queued ANC run.")
    finally:
        np.save(CONTROL_WEIGHTS_FILE, anc.control_weights)
        print(f"Saved control weights to: {CONTROL_WEIGHTS_FILE} (will resume from these next run)")
        log_file.close()
        print(f"Readings saved to: {log_path}")


if __name__ == "__main__":
    main()
