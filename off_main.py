import numpy as np
import sounddevice as sd
from offline_cancellation import OfflineAdaptiveFilter as off
# Audio settings
SAMPLE_RATE = 48000
DURATION = 1.0  # seconds
BLOCKSIZE = int(SAMPLE_RATE * DURATION)
anc = off(filter_length=128, learning_rate=0.0001)
mic1 = sd.InputStream(device=21, channels=1, samplerate=SAMPLE_RATE)
mic2 = sd.InputStream(device=19, channels=1, samplerate=SAMPLE_RATE)
mic1.start()
mic2.start()

print("Reading audio...")
try:
    while True:
        reference_signal, _ = mic1.read(1024)
        target_signal, _ = mic2.read(1024)
        reference_signal = reference_signal.flatten()
        target_signal = target_signal.flatten()
        for ref_sample, tgt_sample in zip(reference_signal, target_signal):
            predicted, error = anc.process_sample(ref_sample, tgt_sample)
        

except KeyboardInterrupt:
    print("Stopping...")

    mic1.stop()
    mic2.stop()

    mic1.close()
    mic2.close()

    print("Closed successfully")


print(f"Mean error: {np.mean(error):.4f}")
print(f"Mean target signal power: {np.mean(target_signal**2):.8f}")
print(f"Mean reference signal power: {np.mean(reference_signal**2):.8f}")
print(f"Correlation: {np.corrcoef(reference_signal, target_signal)[0, 1]:.4f}")
corrs = []

for shift in range(-50, 50):
    if shift < 0:
        c = np.corrcoef(reference_signal[:shift], target_signal[-shift:])[0,1]
    elif shift > 0:
        c = np.corrcoef(reference_signal[shift:], target_signal[:-shift])[0,1]
    else:
        c = np.corrcoef(reference_signal, target_signal)[0,1]

    corrs.append(c)

print(max(corrs))

