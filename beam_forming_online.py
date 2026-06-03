"""
online_beamforming_realtime.py

Independent real-time beamforming / direction-estimation script.
This file does NOT import or call anything from the offline beamforming code.

Recommended Raspberry Pi 5 + two INMP441 setup:
    - Both INMP441 microphones appear as ONE stereo I2S input device.
    - Left channel  = Mic 1
    - Right channel = Mic 2

This script can also run in a fallback "two_devices" mode, but one stereo device is
strongly preferred because the two channels are then sampled synchronously.
"""

from dataclasses import dataclass
from collections import deque
import queue
import time

import numpy as np
import sounddevice as sd


# =============================================================================
# USER SETTINGS
# =============================================================================

SAMPLE_RATE = 48000
BLOCKSIZE = 1024

# Your Raspberry Pi showed device 0 as the available 2-input sound card.
STEREO_INPUT_DEVICE = 0

# Distance between the two microphones in metres.
# Change this if your actual mic spacing is different.
MIC_SPACING_M = 0.225

# Measured delay when the source is exactly centred at 0 degrees.
# Start with 0.0, then replace it after testing.
CALIBRATION_DELAY_OFFSET = 0.5


SPEED_OF_SOUND = 343.0
PRINT_EVERY_SECONDS = 0.25

# Correlation sanity limit.
# A real physical delay cannot be larger than mic_spacing / speed_of_sound.
# We add a small safety margin.
MAX_DELAY_MARGIN_SAMPLES = 0
MIN_RMS_FOR_DIRECTION = 0.003
MIN_CONFIDENCE_FOR_DIRECTION = 5.0

# =============================================================================
# GEOMETRY
# =============================================================================

@dataclass
class TwoMicGeometry:
    mic_spacing_m: float
    speed_of_sound: float = 343.0

    def max_physical_delay_samples(self, sample_rate: int) -> int:
        max_delay_seconds = self.mic_spacing_m / self.speed_of_sound
        return int(np.ceil(max_delay_seconds * sample_rate)) + MAX_DELAY_MARGIN_SAMPLES

    def delay_to_angle_deg(self, delay_samples: float, sample_rate: int) -> float:
        delay_seconds = delay_samples / sample_rate
        sin_theta = (delay_seconds * self.speed_of_sound) / self.mic_spacing_m
        sin_theta = np.clip(sin_theta, -1.0, 1.0)
        return float(np.degrees(np.arcsin(sin_theta)))


# =============================================================================
# REAL-TIME BEAMFORMER
# =============================================================================

class RealTimeBeamformer:
    def __init__(self, geometry: TwoMicGeometry, sample_rate: int):
        self.geometry = geometry
        self.sample_rate = sample_rate
        self.max_lag = geometry.max_physical_delay_samples(sample_rate)

    @staticmethod
    def split_stereo_block(stereo_block):
        block = np.asarray(stereo_block, dtype=np.float32)

        if block.ndim != 2:
            raise ValueError(f"Expected stereo block with shape (samples, channels), got {block.shape}")

        if block.shape[1] < 2:
            raise ValueError(
                f"Expected 2 input channels, but got {block.shape[1]}. "
                "Your two INMP441 microphones should appear as one stereo input device."
            )

        mic1 = block[:, 0]
        mic2 = block[:, 1]
        return mic1, mic2

    @staticmethod
    def remove_dc(signal):
        return signal - np.mean(signal)

    def estimate_delay_samples(self, mic1, mic2):
        """
        GCC-PHAT delay estimation with sub-sample interpolation.

        Positive delay:
            mic2 is delayed relative to mic1

        Negative delay:
            mic1 is delayed relative to mic2
        """
        mic1 = self.remove_dc(mic1)
        mic2 = self.remove_dc(mic2)

        # Apply a window to reduce edge artefacts
        window = np.hanning(len(mic1))
        mic1 = mic1 * window
        mic2 = mic2 * window

        n = 1
        while n < len(mic1) + len(mic2):
            n *= 2

        X1 = np.fft.rfft(mic1, n=n)
        X2 = np.fft.rfft(mic2, n=n)

        cross_spectrum = X2 * np.conj(X1)
        cross_spectrum /= np.abs(cross_spectrum) + 1e-12

        corr = np.fft.irfft(cross_spectrum, n=n)

        max_lag = self.max_lag

        # Put negative lags on the left and positive lags on the right
        corr = np.concatenate((corr[-max_lag:], corr[:max_lag + 1]))
        lags = np.arange(-max_lag, max_lag + 1)

        peak_index = int(np.argmax(corr))
        delay = float(lags[peak_index])

        # Sub-sample parabolic interpolation
        if 0 < peak_index < len(corr) - 1:
            y0 = corr[peak_index - 1]
            y1 = corr[peak_index]
            y2 = corr[peak_index + 1]

            denominator = y0 - 2 * y1 + y2
            if abs(denominator) > 1e-12:
                correction = 0.5 * (y0 - y2) / denominator
                delay += correction

        confidence = corr[peak_index] / (np.mean(np.abs(corr)) + 1e-12)

        return delay, confidence

    @staticmethod
    def apply_integer_delay(signal, delay_samples: int):
        delay_samples = int(delay_samples)
        shifted = np.roll(signal, delay_samples)

        if delay_samples > 0:
            shifted[:delay_samples] = 0
        elif delay_samples < 0:
            shifted[delay_samples:] = 0

        return shifted

    def delay_and_sum(self, mic1, mic2, delay_samples: int):
        mic2_aligned = self.apply_integer_delay(mic2, -delay_samples)
        return 0.5 * mic1 + 0.5 * mic2_aligned

    @staticmethod
    def rms(signal) -> float:
        return float(np.sqrt(np.mean(signal ** 2)))

    @staticmethod
    def peak(signal) -> float:
        return float(np.max(np.abs(signal)))

    def process_stereo_block(self, stereo_block) -> dict:
        mic1, mic2 = self.split_stereo_block(stereo_block)

        mic1_rms = self.rms(mic1)
        mic2_rms = self.rms(mic2)

        # If the input is basically silence, do not estimate direction.
        # Otherwise np.argmax can choose the first lag, which is -max_lag.
        if mic1_rms < MIN_RMS_FOR_DIRECTION or mic2_rms < MIN_RMS_FOR_DIRECTION:
            return {
                "delay_samples": 0.0,
                "corrected_delay_samples": 0.0,
                "delay_seconds": 0.0,
                "angle_deg": None,
                "confidence": 0.0,
                "mic1_rms": mic1_rms,
                "mic2_rms": mic2_rms,
                "mic1_peak": self.peak(mic1),
                "mic2_peak": self.peak(mic2),
                "beamformed_rms": 0.0,
                "valid": False,
            }

        delay_samples, confidence = self.estimate_delay_samples(mic1, mic2)

        corrected_delay_samples = delay_samples - CALIBRATION_DELAY_OFFSET

        delay_seconds = corrected_delay_samples / self.sample_rate
        angle_deg = self.geometry.delay_to_angle_deg(corrected_delay_samples, self.sample_rate)

        beamformed = self.delay_and_sum(mic1, mic2, int(round(corrected_delay_samples)))

        return {
            "delay_samples": delay_samples,
            "corrected_delay_samples": corrected_delay_samples,
            "delay_seconds": delay_seconds,
            "angle_deg": angle_deg,
            "confidence": confidence,
            "mic1_rms": mic1_rms,
            "mic2_rms": mic2_rms,
            "mic1_peak": self.peak(mic1),
            "mic2_peak": self.peak(mic2),
            "beamformed_rms": self.rms(beamformed),
            "valid": confidence >= MIN_CONFIDENCE_FOR_DIRECTION,
        }


# =============================================================================
# STREAM RUNNER
# =============================================================================

class StereoInputRunner:
    def __init__(
        self,
        beamformer: RealTimeBeamformer,
        device: int,
        sample_rate: int,
        blocksize: int,
        print_every_seconds: float,
    ):
        self.beamformer = beamformer
        self.angle_history = deque(maxlen=7)
        self.device = device
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.print_every_seconds = print_every_seconds
        self.audio_queue = queue.Queue(maxsize=8)
        self.last_print_time = 0.0

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"\nAudio status: {status}")

        try:
            self.audio_queue.put_nowait(indata.copy())
        except queue.Full:
            # If processing falls behind, drop the oldest block rather than freezing.
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                pass
            self.audio_queue.put_nowait(indata.copy())

    def print_header(self):
        print("=" * 70)
        print("REAL-TIME TWO-MIC BEAMFORMING / DIRECTION ESTIMATION")
        print("=" * 70)
        print(f"Input device       : {self.device}")
        print(f"Sample rate        : {self.sample_rate} Hz")
        print(f"Blocksize          : {self.blocksize} samples")
        print(f"Mic spacing        : {self.beamformer.geometry.mic_spacing_m:.3f} m")
        print(f"Max physical lag   : ±{self.beamformer.max_lag} samples")
        print("Channel mapping    : channel 0 = Mic 1, channel 1 = Mic 2")
        print("Press Ctrl+C to stop.")
        print("=" * 70)

    def run(self):
        self.print_header()

        with sd.InputStream(
            device=self.device,
            channels=2,
            samplerate=self.sample_rate,
            blocksize=self.blocksize,
            dtype="float32",
            callback=self.audio_callback,
        ):
            while True:
                stereo_block = self.audio_queue.get()
                results = self.beamformer.process_stereo_block(stereo_block)
                self.print_results(results)

    def print_results(self, results: dict):
        now = time.time()
        if now - self.last_print_time < self.print_every_seconds:
            return

        self.last_print_time = now

        if results["valid"] and results["angle_deg"] is not None:
            self.angle_history.append(results["angle_deg"])

        if len(self.angle_history) > 0:
            smooth_angle = np.median(self.angle_history)
            angle_text = f"{smooth_angle:>7.2f}°"
        else:
            angle_text = "waiting"

        print(
            f"Delay: {results['delay_samples']:>4} samples "
            f"({results['delay_seconds'] * 1000:>7.3f} ms) | "
            f"Angle: {angle_text} | "
            f"RMS L/R: {results['mic1_rms']:.4f}/{results['mic2_rms']:.4f} | "
            f"Beam RMS: {results['beamformed_rms']:.4f} | "
            f"Confidence: {results['confidence']:.4f}",
            end="\r",
        )


# =============================================================================
# MAIN
# =============================================================================

def print_available_devices():
    print("\nAvailable audio devices:\n")
    print(sd.query_devices())
    print()


def main():
    print_available_devices()

    geometry = TwoMicGeometry(
        mic_spacing_m=MIC_SPACING_M,
        speed_of_sound=SPEED_OF_SOUND,
    )

    beamformer = RealTimeBeamformer(
        geometry=geometry,
        sample_rate=SAMPLE_RATE,
    )

    runner = StereoInputRunner(
        beamformer=beamformer,
        device=STEREO_INPUT_DEVICE,
        sample_rate=SAMPLE_RATE,
        blocksize=BLOCKSIZE,
        print_every_seconds=PRINT_EVERY_SECONDS,
    )

    try:
        runner.run()
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as exc:
        print("\nERROR:")
        print(exc)
        print("\nMost likely checks:")
        print("1. Confirm the device number with: python3 -m sounddevice")
        print("2. Your screenshot showed device 0, so STEREO_INPUT_DEVICE should be 0.")
        print("3. Confirm the device has 2 input channels.")
        print("4. Check the INMP441 wiring: both SD pins to Pi GPIO20, one L/R to GND, the other L/R to 3.3V.")


if __name__ == "__main__":
    main()
