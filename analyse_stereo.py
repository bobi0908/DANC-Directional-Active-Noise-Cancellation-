import numpy as np
from scipy.io import wavfile

sr, audio = wavfile.read("stereo_test.wav")

print("Sample rate:", sr)
print("Shape:", audio.shape)
print("Dtype:", audio.dtype)

audio = np.asarray(audio)

if np.issubdtype(audio.dtype, np.integer):
    audio_float = audio.astype(np.float32) / np.iinfo(audio.dtype).max
else:
    audio_float = audio.astype(np.float32)

if audio_float.ndim == 1:
    print("Recording is mono, not stereo.")
    print("Peak:", np.max(np.abs(audio_float)))
    print("RMS:", np.sqrt(np.mean(audio_float ** 2)))
else:
    for ch in range(audio_float.shape[1]):
        x = audio_float[:, ch]
        print(f"\nChannel {ch}:")
        print("  Peak:", np.max(np.abs(x)))
        print("  RMS:", np.sqrt(np.mean(x ** 2)))
        print("  Power:", np.mean(x ** 2))

    half = len(audio_float) // 2

    print("\nFirst half RMS:")
    print("  Channel 0:", np.sqrt(np.mean(audio_float[:half, 0] ** 2)))
    print("  Channel 1:", np.sqrt(np.mean(audio_float[:half, 1] ** 2)))

    print("\nSecond half RMS:")
    print("  Channel 0:", np.sqrt(np.mean(audio_float[half:, 0] ** 2)))
    print("  Channel 1:", np.sqrt(np.mean(audio_float[half:, 1] ** 2)))
