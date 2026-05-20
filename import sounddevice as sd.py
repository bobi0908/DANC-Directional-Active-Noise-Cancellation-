import sounddevice as sd
import numpy as np
audio = sd.rec(48000, samplerate=48000, channels=1)
sd.wait()
sd.default.device = (None,19)
print(sd.default.device)


print("MAX:", np.max(np.abs(audio)))
print("MEAN:", np.mean(np.abs(audio)))