"""
Direct-callback FxLMS ANC -- the timing-correct version.

Why this exists
---------------
actual_anc_queue_run.py decouples audio I/O from processing with a queue: the
callback enqueues the input and plays an output block that was computed one (or
more) callbacks earlier. That ~1-block (~85 ms at BLOCK_SIZE=4096) delay is NOT
captured by the secondary-path calibration, so the anti-noise comes out at the
wrong phase and ADDS to the noise instead of cancelling it.

This script removes the queue. The FxLMS for a whole block runs in C
(libfxlms.so, reused from actual_anc_queue_run.py) in microseconds, so it runs
*inside* the audio callback and outputs the anti-noise in the SAME block -- no
delay, timing consistent with calibration. All disk/terminal I/O is done on a
separate thread so the callback stays real-time-safe.

It keeps everything the exhibition needs:
  * writes the same anc_log_*.csv -> live_anc_graph.py works unchanged
  * reads anc_control.txt -> the graph's toggle button AND the ENTER key work

Honest scope
------------
  * Even with perfect timing, the ~43 ms electronic latency means only a
    PERIODIC source (a tone, e.g. 175 Hz) can be cancelled, not broadband.
  * The secondary path must match THIS script's audio path. Recalibrate with
    actual_anc_fxlms.py (MODE="calibrate") in the final rig before running. If
    it still adds, a calibration done through this exact Stream is the next step.

Usage
-----
  1. (Re)calibrate the secondary path in the box:  actual_anc_fxlms.py, MODE="calibrate"
  2. Play a steady 175 Hz tone as the noise source.
  3. Run:  .venv/bin/python actual_anc_direct.py
  4. (Optional) live graph in another terminal:  DISPLAY=:0 python live_anc_graph.py
                                       or remote:  ANC_GRAPH_BACKEND=WebAgg python live_anc_graph.py
  Press Ctrl+C to stop. Press ENTER (or the graph button) to toggle ANC on/off.
"""

import csv
import os
import sys
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd

# Reuse the fast, already-working C-library FxLMS wrapper. Importing it loads
# libfxlms.so once; it does not start the queued run (that's guarded by
# __main__). Everything else below is self-contained and independent.
from actual_anc_queue_run import OnlineFxLMSFilter


# =========================================================
# AUDIO SETTINGS  (match actual_anc_queue_run.py)
# =========================================================

SAMPLE_RATE = 48000
DEVICE_ID = 0

INPUT_CHANNELS = 2
OUTPUT_CHANNELS = 2

ERROR_CHANNEL = 0       # inside box / cancellation point mic
REFERENCE_CHANNEL = 1   # outside / source-side mic

BLOCK_SIZE = 4096

# Dedicated secondary path measured through THIS script's own audio path
# (MODE="calibrate" below). Falls back to the old secondary_path.npy if it
# hasn't been measured yet.
SECONDARY_PATH_FILE = "secondary_path_direct.npy"
SECONDARY_PATH_FALLBACK = "secondary_path.npy"
# Separate weights file so this script never inherits the queued run's
# wrong-timing weights (and vice-versa).
CONTROL_WEIGHTS_FILE = "control_weights_direct.npy"


# =========================================================
# RUN SETTINGS
# =========================================================

ANC_OUTPUT_LIMIT_INT16 = 10000
MAX_COMMAND = 0.02

CONTROL_FILTER_LENGTH = 16

SECONDARY_WINDOW_BEFORE = 20
SECONDARY_WINDOW_AFTER = 150

LEARNING_RATE = 3e-6
EPSILON = 1e-6
LEAKAGE = 0
UPDATE_SIGN = -1.0

IGNORE_START_SECONDS = 1.0
BASELINE_SECONDS = 8.0
PRINT_INTERVAL_SECONDS = 0.25

MIN_ERROR_RMS_FOR_ADAPTATION = 0.0003

ENABLE_SAFETY_MUTE = True
SAFETY_FACTOR = 25.0

# Live demo on/off toggle. Starts ON here so a test immediately shows whether
# the filter builds output and cancels; set False for the "reveal" demo flow.
ENABLE_ANC_TOGGLE = True
START_WITH_ANC_ON = False

# Same control file the live graph writes, so its toggle button drives this run.
ANC_CONTROL_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "anc_control.txt"
)


# =========================================================
# MODE + CALIBRATION SETTINGS
# =========================================================

# "calibrate": measure the secondary path through THIS script's own Stream
#   (so its latency/timing matches the run exactly), then set MODE="run".
# "run": do the live ANC.
MODE = "run"

CALIB_SECONDS = 12.0          # length of the calibration noise burst
CALIB_SEED = 123              # fixed so the excitation is reproducible
CALIB_COMMAND_AMP = 0.4       # float command peak (played at x ANC_OUTPUT_LIMIT_INT16)
CALIB_SAVE_LENGTH = 16384     # impulse-response samples to keep (~340 ms, covers the delay)
CALIB_REG = 1e-2             # deconvolution regularization (fraction of mean power)
CALIB_GUARD_SAMPLES = 48      # zero the first ~1 ms (kills lag-0 electrical feedthrough)


# =========================================================
# SHARED STATE  (callback writes, logger thread reads)
# =========================================================

state = {
    "callback_blocks": 0,
    "processed_blocks": 0,
    "portaudio_status_count": 0,
    "last_portaudio_status": "",

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

    "anc_enabled": True,
}

# Filled by the callback, drained to disk by the main thread (never write files
# from inside the audio callback -- that would risk audio glitches).
log_buffer = deque(maxlen=4000)


# =========================================================
# HELPERS
# =========================================================

def rms(x):
    x = np.asarray(x, dtype=np.float32).ravel()
    return float(np.sqrt(np.mean(x ** 2)))


def power(x):
    x = np.asarray(x, dtype=np.float32).ravel()
    return float(np.mean(x ** 2))


def write_anc_control(enabled):
    """Atomically record the requested ANC on/off state."""
    try:
        tmp = ANC_CONTROL_FILE + ".tmp"
        with open(tmp, "w") as f:
            f.write("on" if enabled else "off")
        os.replace(tmp, ANC_CONTROL_FILE)
    except Exception:
        pass


def read_anc_control(default):
    """Requested on/off state from the control file, or `default` if missing."""
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
    """Each ENTER flips ANC on/off via the shared control file (daemon thread)."""
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
# CALIBRATION (measured through THIS script's own Stream)
# =========================================================

def identify_ir(played, recorded, reg=CALIB_REG):
    """Least-squares FIR estimate of the path played -> recorded, via FFT
    (Wiener deconvolution). The path delay shows up as the location of the
    main peak. Noise uncorrelated with `played` is suppressed automatically.
    """
    n = len(played)
    X = np.fft.rfft(played)
    Y = np.fft.rfft(recorded)
    power_x = np.abs(X) ** 2
    H = (Y * np.conj(X)) / (power_x + reg * np.mean(power_x))
    h = np.fft.irfft(H, n=n)
    return h.astype(np.float32)


def calibrate_secondary_path():
    """Play a known noise burst and record the error mic THROUGH the same
    Stream/blocksize/latency the run uses, then identify speaker->error-mic.

    Because it goes through the identical audio path, the measured delay and
    gain match the run exactly -- which is the whole point (no playrec-vs-Stream
    or 8000-vs-10000 mismatch).
    """
    n = int(CALIB_SECONDS * SAMPLE_RATE)
    rng = np.random.default_rng(CALIB_SEED)
    excitation = rng.normal(0.0, 1.0, n).astype(np.float32)
    peak = float(np.max(np.abs(excitation)))
    if peak > 0:
        excitation = excitation / peak
    excitation = (CALIB_COMMAND_AMP * excitation).astype(np.float32)

    played = np.zeros(n, dtype=np.float32)
    recorded = np.zeros(n, dtype=np.float32)
    pos = {"w": 0}
    done = threading.Event()

    def cb(indata, outdata, frames, time_info, status):
        outdata[:] = 0
        w = pos["w"]
        if w >= n:
            done.set()
            return
        end = min(w + frames, n)
        m = end - w
        cmd = excitation[w:end]
        out_int16 = (cmd * ANC_OUTPUT_LIMIT_INT16).astype(np.int16)
        outdata[:m, 0] = out_int16
        outdata[:m, 1] = out_int16
        recorded[w:end] = np.asarray(indata, dtype=np.float32)[:m, ERROR_CHANNEL]
        played[w:end] = cmd
        pos["w"] = end
        if end >= n:
            done.set()

    print("\nSecondary-path calibration THROUGH this script's own audio path.")
    print("Keep the room QUIET; speaker + error mic in their FINAL positions.")
    print(f"Playing {CALIB_SECONDS:.0f}s of calibration noise...")

    with sd.Stream(
        device=DEVICE_ID,
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=(INPUT_CHANNELS, OUTPUT_CHANNELS),
        dtype=("float32", "int16"),
        callback=cb,
        latency="high",
    ):
        done.wait(timeout=CALIB_SECONDS + 5.0)

    w = pos["w"]
    start = min(int(IGNORE_START_SECONDS * SAMPLE_RATE), w)
    played = played[start:w]
    recorded = recorded[start:w]
    if len(played) < SAMPLE_RATE:
        print("ERROR: captured too little audio -- is the device free and working?")
        return

    print(f"Captured {len(played) / SAMPLE_RATE:.1f}s "
          f"(command RMS={rms(played):.5f}, error mic RMS={rms(recorded):.5f}).")
    print("Identifying secondary path...")

    h = identify_ir(played, recorded, reg=CALIB_REG)
    secondary_path = h[:CALIB_SAVE_LENGTH].copy()
    secondary_path[:CALIB_GUARD_SAMPLES] = 0.0  # remove lag-0 electrical feedthrough
    np.save(SECONDARY_PATH_FILE, secondary_path.astype(np.float32))

    pk = int(np.argmax(np.abs(secondary_path)))
    win = secondary_path[max(0, pk - SECONDARY_WINDOW_BEFORE):pk + SECONDARY_WINDOW_AFTER]
    energy_frac = 100.0 * float(np.sum(win ** 2)) / (float(np.sum(secondary_path ** 2)) + 1e-20)

    print(f"\nSaved {SECONDARY_PATH_FILE} (length {len(secondary_path)}).")
    print(f"  peak tap {pk} ({pk / SAMPLE_RATE * 1000:.1f} ms), value {secondary_path[pk]:+.3f}")
    print(f"  peak/rms ratio {np.max(np.abs(secondary_path)) / (rms(secondary_path) + 1e-20):.1f} "
          f"(impulse-like if >> 1)")
    print(f"  energy inside the run's window: {energy_frac:.0f}%")
    if pk < CALIB_GUARD_SAMPLES + 5:
        print("  WARNING: peak is very early -- likely electrical feedthrough, not acoustic.")
    if pk > CALIB_SAVE_LENGTH - SECONDARY_WINDOW_AFTER:
        print("  WARNING: peak near the end -- increase CALIB_SAVE_LENGTH and recalibrate.")
    print('\nDone. Set MODE = "run" and start the demo.')


# =========================================================
# MAIN
# =========================================================

def main():
    sp_file = SECONDARY_PATH_FILE
    if not os.path.exists(sp_file):
        if os.path.exists(SECONDARY_PATH_FALLBACK):
            print(f"\n(No {SECONDARY_PATH_FILE} yet -- using {SECONDARY_PATH_FALLBACK}.")
            print(' For correct timing, run with MODE="calibrate" first.)')
            sp_file = SECONDARY_PATH_FALLBACK
        else:
            raise FileNotFoundError(
                f"No secondary path found. Run with MODE='calibrate' first "
                f"(no {SECONDARY_PATH_FILE} or {SECONDARY_PATH_FALLBACK})."
            )

    full_secondary_path = np.load(sp_file)
    secondary_peak_index = int(np.argmax(np.abs(full_secondary_path)))
    secondary_start = max(0, secondary_peak_index - SECONDARY_WINDOW_BEFORE)
    secondary_end = min(len(full_secondary_path),
                        secondary_peak_index + SECONDARY_WINDOW_AFTER)
    secondary_path = full_secondary_path[secondary_start:secondary_end]

    print("\nLoaded secondary path:", sp_file)
    print("Strongest tap:", secondary_peak_index,
          f"({secondary_peak_index / SAMPLE_RATE * 1000:.1f} ms)")
    print("Window:", secondary_start, "->", secondary_end,
          "(len", len(secondary_path), ")")

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
        loaded = np.load(CONTROL_WEIGHTS_FILE)
        if len(loaded) == anc.control_length:
            saved_weights = loaded.astype(np.float32)
            print(f"\nResuming control weights from {CONTROL_WEIGHTS_FILE}.")
        else:
            print(f"\nIgnoring {CONTROL_WEIGHTS_FILE} (wrong length); starting from zero.")

    ignore_blocks = int(np.ceil(IGNORE_START_SECONDS * SAMPLE_RATE / BLOCK_SIZE))
    baseline_blocks = int(np.ceil(BASELINE_SECONDS * SAMPLE_RATE / BLOCK_SIZE))

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

    print("\nDirect-callback ANC (FxLMS runs IN the callback -- no queue delay).")
    print("Press Ctrl+C to stop.")
    if ENABLE_ANC_TOGGLE:
        print("Press ENTER (or the graph button) to toggle ANC ON/OFF "
              "(wait for 'ANC running' first).")
    print(f"Logging to: {log_path}\n")

    state["anc_enabled"] = True
    if ENABLE_ANC_TOGGLE:
        state["anc_enabled"] = START_WITH_ANC_ON
        write_anc_control(state["anc_enabled"])
        threading.Thread(target=anc_toggle_listener, daemon=True).start()

    run_start = time.time()

    def audio_callback(indata, outdata, frames, time_info, status):
        state["callback_blocks"] += 1
        if status:
            state["portaudio_status_count"] += 1
            state["last_portaudio_status"] = str(status)
        else:
            state["last_portaudio_status"] = ""

        outdata[:] = 0  # default to silence

        block = np.asarray(indata, dtype=np.float32)
        reference_block = block[:, REFERENCE_CHANNEL]
        error_block = block[:, ERROR_CHANNEL]
        reference_rms = rms(reference_block)
        error_rms = rms(error_block)

        state["processed_blocks"] += 1
        pb = state["processed_blocks"]

        # Startup ignore period.
        if pb <= ignore_blocks:
            state["latest_status"] = "ignoring startup"
            return

        # Baseline period: speaker stays silent, measure uncancelled error.
        if pb <= ignore_blocks + baseline_blocks:
            state["baseline_power_sum"] += power(error_block)
            state["baseline_count"] += 1
            state["latest_ref_rms"] = reference_rms
            state["latest_err_rms"] = error_rms
            state["latest_out_max"] = 0
            state["latest_adapt"] = False
            state["latest_status"] = "baseline"
            return

        # Finalize baseline once, then engage.
        if state["baseline_power"] is None:
            state["baseline_power"] = (
                state["baseline_power_sum"] / max(state["baseline_count"], 1)
            )
            state["baseline_rms"] = float(np.sqrt(state["baseline_power"]))
            anc.reset()
            if saved_weights is not None:
                anc.control_weights[:] = saved_weights

        adapt = state["anc_enabled"] and error_rms >= MIN_ERROR_RMS_FOR_ADAPTATION

        command_block = anc.process_block(reference_block, error_block, adapt=adapt)
        command_block = np.clip(command_block, -MAX_COMMAND, MAX_COMMAND)
        output_mono = (command_block * ANC_OUTPUT_LIMIT_INT16).astype(np.int16)

        error_power_now = power(error_block)

        if not state["anc_enabled"]:
            state["latest_status"] = "ANC OFF"          # outdata already silent
            adapt = False
            out_max = 0
        elif (ENABLE_SAFETY_MUTE and state["baseline_power"] is not None
              and error_power_now > SAFETY_FACTOR * state["baseline_power"]):
            state["latest_status"] = "SAFETY MUTE"       # outdata already silent
            adapt = False
            out_max = 0
        else:
            outdata[:, 0] = output_mono
            outdata[:, 1] = output_mono
            state["latest_status"] = "ANC running"
            out_max = int(np.max(np.abs(output_mono))) if len(output_mono) else 0

        if state["baseline_power"] and error_power_now > 0:
            erle = 10.0 * np.log10(state["baseline_power"] / error_power_now)
        else:
            erle = np.nan

        state["latest_erle"] = erle
        state["latest_ref_rms"] = reference_rms
        state["latest_err_rms"] = error_rms
        state["latest_out_max"] = out_max
        state["latest_adapt"] = adapt

        # Hand the row to the logger thread (no file I/O in the callback).
        log_buffer.append((
            time.time() - run_start, erle, state["baseline_rms"],
            reference_rms, error_rms, out_max, int(adapt), state["latest_status"],
        ))

    last_print = 0.0
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
                if ENABLE_ANC_TOGGLE:
                    state["anc_enabled"] = read_anc_control(state["anc_enabled"])

                # Drain logged rows to disk.
                wrote = False
                while log_buffer:
                    t, erle, base, ref_rms, err_rms, out_max, adapt, status = log_buffer.popleft()
                    log_writer.writerow([
                        f"{t:.3f}",
                        f"{erle:.4f}" if not np.isnan(erle) else "",
                        f"{base:.6f}" if base is not None else "",
                        f"{ref_rms:.6f}",
                        f"{err_rms:.6f}",
                        out_max, adapt, status,
                    ])
                    wrote = True
                if wrote:
                    log_file.flush()

                now = time.time()
                if now - last_print >= PRINT_INTERVAL_SECONDS:
                    last_print = now
                    adapt_text = "ADAPT" if state["latest_adapt"] else "NOADAPT"
                    line = (
                        f"ERLE={state['latest_erle']:6.2f}dB | "
                        f"REF={state['latest_ref_rms']:.4f} | "
                        f"ERR={state['latest_err_rms']:.4f} | "
                        f"OUT={state['latest_out_max']:3d} | "
                        f"{adapt_text:7s} | "
                        f"{state['latest_status'][:16]}"
                    )
                    print(f"\r{line:<76}", end="", flush=True)

                time.sleep(0.02)

    except KeyboardInterrupt:
        print("\nStopped direct ANC run.")
    finally:
        np.save(CONTROL_WEIGHTS_FILE, anc.control_weights)
        print(f"Saved control weights to: {CONTROL_WEIGHTS_FILE}")
        log_file.close()
        print(f"Readings saved to: {log_path}")


if __name__ == "__main__":
    # Mode can be overridden on the command line (e.g.
    # `... actual_anc_direct.py calibrate`); with no argument it uses MODE above.
    mode = sys.argv[1] if len(sys.argv) > 1 else MODE
    print("Using device:")
    print(sd.query_devices(DEVICE_ID))
    if mode == "calibrate":
        calibrate_secondary_path()
    elif mode == "run":
        main()
    else:
        raise ValueError(f'mode must be "calibrate" or "run" (got "{mode}").')
