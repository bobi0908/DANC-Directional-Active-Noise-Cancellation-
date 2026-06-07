import time
import numpy as np
import sounddevice as sd

SAMPLE_RATE = 48000
DEVICE_ID = 0
INPUT_CHANNELS = 2
OUTPUT_CHANNELS = 2

# Try 4096 first. If unstable, try 8192.
BLOCK_SIZE = 4096

state = {
    "blocks": 0,
    "status_count": 0,
    "last_status": "",
    "ch0_rms": 0.0,
    "ch1_rms": 0.0,
}

def rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float32) ** 2)))

def callback(indata, outdata, frames, time_info, status):
    state["blocks"] += 1

    if status:
        state["status_count"] += 1
        state["last_status"] = str(status)

    # Output complete silence.
    outdata[:] = 0

    # Tiny amount of input analysis only.
    state["ch0_rms"] = rms(indata[:, 0])
    state["ch1_rms"] = rms(indata[:, 1])

print(sd.query_devices(DEVICE_ID))
print("Starting minimal duplex test: input + silent output.")
print("Press Ctrl+C to stop.\n")

try:
    with sd.Stream(
        device=DEVICE_ID,
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=(INPUT_CHANNELS, OUTPUT_CHANNELS),
        dtype=("float32", "int16"),
        latency="high",
        callback=callback
    ):
        while True:
            print(
                f"\rblocks={state['blocks']} | "
                f"status_count={state['status_count']} | "
                f"ch0_rms={state['ch0_rms']:.6f} | "
                f"ch1_rms={state['ch1_rms']:.6f} | "
                f"last_status={state['last_status'][:40]}",
                end="",
                flush=True
            )
            time.sleep(0.25)

except KeyboardInterrupt:
    print("\nStopped.")
