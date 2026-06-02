import time
import numpy as np
import sounddevice as sd
from scipy.signal import butter, sosfilt
from offline_cancellation import OnlineNLMSFilter


# =========================================================
# SETTINGS
# =========================================================

SAMPLE_RATE = 48000
DEVICE_ID = 0
CHANNELS = 2

REFERENCE_CHANNEL = 0
TARGET_CHANNEL = 1

BLOCK_SIZE = 1024

FILTER_LENGTH = 256
LEARNING_RATE = 0.02

LOWCUT = 80
HIGHCUT = 1000
USE_BANDPASS = True

IGNORE_START_SECONDS = 1.0

# If target mic is quieter than this, we do not adapt.
# This avoids the filter learning mostly electrical/background noise.
MIN_TARGET_RMS_FOR_ADAPTATION = 0.001

PRINT_INTERVAL_SECONDS = 0.25


# =========================================================
# REAL-TIME BANDPASS FILTER
# =========================================================

class RealTimeBandpass:
    """
    Causal bandpass filter for live audio.

    This replaces sosfiltfilt from the offline code.
    sosfiltfilt uses future samples, so it is not valid for live processing.
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

        # One filter state per channel
        self.zi = np.zeros(
            (channels, self.sos.shape[0], 2),
            dtype=np.float32
        )

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
# HELPERS
# =========================================================

def rms(x):
    x = np.asarray(x, dtype=np.float32).ravel()
    return np.sqrt(np.mean(x ** 2))


def power(x):
    x = np.asarray(x, dtype=np.float32).ravel()
    return np.mean(x ** 2)


def compute_erle(target_block, error_block):
    target_power = power(target_block)
    error_power = power(error_block)

    if target_power <= 0 or error_power <= 0:
        return np.nan

    return 10 * np.log10(target_power / error_power)


def make_bar(value, scale=1000, width=30):
    """
    Simple terminal level bar based on RMS.
    """
    filled = int(min(value * scale, 1.0) * width)
    return "#" * filled + "-" * (width - filled)


# =========================================================
# MAIN ONLINE TEST
# =========================================================

def main():
    print("Available devices:")
    for i, device in enumerate(sd.query_devices()):
        print(
            i,
            device["name"],
            "| inputs:", device["max_input_channels"],
            "| outputs:", device["max_output_channels"],
            "| default SR:", device["default_samplerate"]
        )

    print("\nUsing device:")
    print(sd.query_devices(DEVICE_ID))

    anc = OnlineNLMSFilter(
        filter_length=FILTER_LENGTH,
        learning_rate=LEARNING_RATE
    )

    if USE_BANDPASS:
        bandpass_filter = RealTimeBandpass(
            sample_rate=SAMPLE_RATE,
            low=LOWCUT,
            high=HIGHCUT,
            channels=CHANNELS
        )
    else:
        bandpass_filter = None

    ignore_blocks = int(
        IGNORE_START_SECONDS * SAMPLE_RATE / BLOCK_SIZE
    )

    block_counter = 0
    last_print_time = 0

    print("\nStarting online mic-only adaptive test.")
    print("No speaker output yet.")
    print("Play pink noise / low-frequency noise.")
    print("Press Ctrl+C to stop.\n")

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

                block_counter += 1

                # Ignore startup transient.
                # We also reset the adaptive filter after this period,
                # so it does not learn the startup click.
                if block_counter <= ignore_blocks:
                    if block_counter == ignore_blocks:
                        anc.reset()
                        print("Startup ignored. Adaptive filter reset. Now learning...\n")
                    continue

                if USE_BANDPASS:
                    block = bandpass_filter.process(block)

                reference_block = block[:, REFERENCE_CHANNEL]
                target_block = block[:, TARGET_CHANNEL]

                target_rms = rms(target_block)
                reference_rms = rms(reference_block)

                adapt = target_rms >= MIN_TARGET_RMS_FOR_ADAPTATION

                predicted_block, error_block = anc.process_block(
                    reference_block,
                    target_block,
                    adapt=adapt
                )

                erle = compute_erle(target_block, error_block)

                now = time.time()

                if now - last_print_time >= PRINT_INTERVAL_SECONDS:
                    last_print_time = now

                    ref_bar = make_bar(reference_rms)
                    tgt_bar = make_bar(target_rms)
                    err_bar = make_bar(rms(error_block))

                    status = "ADAPT" if adapt else "QUIET"

                    overflow_text = " OVERFLOW" if overflowed else ""

                    print(
                        f"\r"
                        f"ERLE={erle:6.2f} dB | "
                        f"REF [{ref_bar}] {reference_rms:.5f} | "
                        f"TGT [{tgt_bar}] {target_rms:.5f} | "
                        f"ERR [{err_bar}] {rms(error_block):.5f} | "
                        f"{status}"
                        f"{overflow_text}",
                        end="",
                        flush=True
                    )

    except KeyboardInterrupt:
        print("\nStopped online test.")


if __name__ == "__main__":
    main()
