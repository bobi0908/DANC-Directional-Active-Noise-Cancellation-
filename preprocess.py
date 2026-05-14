import sounddevice as sd
import numpy as np
import time

import sounddevice as sd
INPUT_CHANNELS = 1      # microphone array (1 for now)


#Take raw chunk of microphone audio 
#return cleaner chunk for analysis ß

import numpy as np

def preprocess(audio_frame):
    audio_frame = audio_frame.astype(np.float32)
    audio_frame = audio_frame - np.mean(audio_frame, axis=0, keepdims=True)

    max_val = np.max(np.abs(audio_frame))

    if max_val > 0:
        audio_frame = audio_frame / max_val * 0.8

    return audio_frame
    











#





#Return clean audio 

