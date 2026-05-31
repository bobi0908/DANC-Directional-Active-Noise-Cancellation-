import numpy as np
import sounddevice as sd
import time

SAMPLE_RATE = 48000
BLOCK_SIZE = 1024
DEVICE_ID = 0      # change this to your sounddevice input device index
CHANNELS = 2


def bar_from_rms(rms, width=40):
    """
    Convert RMS level into a visual bar.

    RMS values are usually small, so this uses a rough scaling.
    Adjust scale if bars are too tiny or always full.
    """
    scale = 80
    level = min(int(rms * scale * width), width)
    return "#" * level + "-" * (width - level)


def analyse_block(block):
    """
    block shape should be (BLOCK_SIZE, 2)
    """
    ch0 = block[:, 0]
    ch1 = block[:, 1]

    rms0 = np.sqrt(np.mean(ch0 ** 2))
    rms1 = np.sqrt(np.mean(ch1 ** 2))

    peak0 = np.max(np.abs(ch0))
    peak1 = np.max(np.abs(ch1))

    return rms0, rms1, peak0, peak1


print("Available devices:")
for i, device in enumerate(sd.query_devices()):
    print(
        i,
        device["name"],
        "| inputs:", device["max_input_channels"],
        "| outputs:", device["max_output_channels"],
        "| default SR:", device["default_samplerate"]
    )

print("\nUsing device:", DEVICE_ID)
print(sd.query_devices(DEVICE_ID))
print("\nLive 2-mic meter. Press Ctrl+C to stop.")
print("Tap/speak near each mic separately and check that different channels respond.\n")

try:
    with sd.InputStream(
        device=DEVICE_ID,
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        dtype="float32"
    ) as stream:

        while True:
            block, overflowed = stream.read(BLOCK_SIZE)

            if overflowed:
                overflow_text = " OVERFLOW"
            else:
                overflow_text = ""

            rms0, rms1, peak0, peak1 = analyse_block(block)

            bar0 = bar_from_rms(rms0)
            bar1 = bar_from_rms(rms1)

            print(
                f"\rCH0 [{bar0}] RMS={rms0:.5f} PEAK={peak0:.5f} | "
                f"CH1 [{bar1}] RMS={rms1:.5f} PEAK={peak1:.5f}"
                f"{overflow_text}",
                end="",
                flush=True
            )

            time.sleep(0.03)

except KeyboardInterrupt:
    print("\nStopped.")
