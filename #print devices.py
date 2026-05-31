#print devices
import sounddevice as sd
import sounddevice as sd

devices = sd.query_devices()

for i, d in enumerate(devices):
    print(
        i,
        d["name"],
        "| inputs:", d["max_input_channels"],
        "| outputs:", d["max_output_channels"],
        "| default samplerate:", d["default_samplerate"]
    )