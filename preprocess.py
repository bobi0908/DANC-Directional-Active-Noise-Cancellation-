import sounddevice as sd
import numpy as np
import time

import sounddevice as sd
INPUT_CHANNELS = 1      # microphone array (1 for now)


#Take raw chunk of microphone audio 
#return cleaner chunk for analysis ß


def preprocess(audio_frame):
    audio_frame = audio_frame.astype(np.float32) #concerts the audio to float32 for processing
    audio_frame = audio_frame - np.mean(audio_frame, axis=0, keepdims=True) #removes DC offset by centering the audio around zero

    max_val = np.max(np.abs(audio_frame)) #finds the maximum absolute value in the audio frame for normalization

    if max_val > 0:
        audio_frame = audio_frame / max_val * 0.8 #normalizes the audio to prevent clipping, scaling it to 80% of the maximum possible value

    return audio_frame












#





#Return clean audio 

