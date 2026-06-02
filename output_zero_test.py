import numpy as np
import sounddevice as sd

SAMPLE_RATE = 48000
DURATION = 2
DEVICE_ID = 0

n = int(DURATION * SAMPLE_RATE)

# Absolute silence
output = np.zeros((n, 2), dtype=np.float32)

print(sd.query_devices(DEVICE_ID))
print("Playing zeros. Speaker should be completely silent.")

sd.play(output, samplerate=SAMPLE_RATE, device=DEVICE_ID, blocking=True)

print("Done.")
