import sounddevice as sd

for i, device in enumerate(sd.query_devices()):
    print(
        i,
        device["name"],
        "| inputs:", device["max_input_channels"],
        "| outputs:", device["max_output_channels"],
        "| default SR:", device["default_samplerate"]
    )
