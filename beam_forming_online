import numpy as np
import sounddevice as sd


SAMPLE_RATE = 48000
BLOCKSIZE = 1024

MIC1_DEVICE = 4
MIC2_DEVICE = 1


def make_1d(signal):
    """Convert a microphone frame to a clean 1D float32 array."""
    return np.asarray(signal, dtype=np.float32).reshape(-1)


def apply_integer_delay(signal, delay_samples):
    """Apply a sample delay using np.roll, then zero the wrapped part."""
    int_delay = int(np.round(delay_samples))
    shifted = np.roll(signal, int_delay)
    if int_delay > 0:
        shifted[:int_delay] = 0
    elif int_delay < 0:
        shifted[int_delay:] = 0
    return shifted


def delay_and_sum_beamformer(mic1, mic2, delay_samples, w1=0.5, w2=0.5):
    """Simple 2-mic delay-and-sum beamformer used in the offline prototype."""
    mic2_aligned = apply_integer_delay(mic2, -delay_samples)
    return w1 * mic1 + w2 * mic2_aligned


def estimate_delay_samples(mic1, mic2):
    """Estimate delay between two channels using cross-correlation."""
    mic1 = mic1 - np.mean(mic1)
    mic2 = mic2 - np.mean(mic2)
    correlation = np.correlate(mic2, mic1, mode='full')
    delay_index = np.argmax(correlation)
    return delay_index - (len(mic1) - 1)


class OnlineDirectionEstimator:
    """Direction-estimation only prototype for live microphone streams."""

    def __init__(self, sample_rate=SAMPLE_RATE, blocksize=BLOCKSIZE):
        self.sample_rate = sample_rate
        self.blocksize = blocksize

    def process_block(self, mic1_block, mic2_block):
        mic1 = make_1d(mic1_block)
        mic2 = make_1d(mic2_block)

        delay_samples = estimate_delay_samples(mic1, mic2)
        delay_seconds = delay_samples / self.sample_rate

        return delay_samples, delay_seconds


def main():
    print("Starting online beamforming + ANC prototype...")
    print("Adjust MIC1_DEVICE / MIC2_DEVICE / OUTPUT_DEVICE if your hardware differs.")

    mic1 = sd.InputStream(
        device=MIC1_DEVICE,
        channels=1,
        samplerate=SAMPLE_RATE,
        blocksize=BLOCKSIZE,
        dtype='float32',
    )
    mic2 = sd.InputStream(
        device=MIC2_DEVICE,
        channels=1,
        samplerate=SAMPLE_RATE,
        blocksize=BLOCKSIZE,
        dtype='float32',
    )
    direction_estimator = OnlineDirectionEstimator(sample_rate=SAMPLE_RATE, blocksize=BLOCKSIZE)

    try:
        mic1.start()
        mic2.start()

        print("Streaming... Press Ctrl+C to stop.")

        while True:
            ref_block, _ = mic1.read(BLOCKSIZE)
            err_block, _ = mic2.read(BLOCKSIZE)

            delay_samples, delay_seconds = direction_estimator.process_block(ref_block, err_block)

            print(f"Estimated delay: {delay_samples} samples ({delay_seconds*1000:.3f} ms)", end='\r')

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        mic1.stop(); mic2.stop()
        mic1.close(); mic2.close()


if __name__ == '__main__':
    main()
