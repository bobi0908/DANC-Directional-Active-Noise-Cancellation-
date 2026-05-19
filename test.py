#framework from gpt
import sounddevice as sd
import numpy as np
import time

# =========================================================
# CONFIG
# =========================================================

SAMPLE_RATE = 44100
BLOCK_SIZE = 256

INPUT_CHANNELS = 1      # microphone array
OUTPUT_CHANNELS = 1     # stereo headphones/speakers

DTYPE = np.float32

# =========================================================
# DEBUG / PERFORMANCE
# =========================================================

callback_count = 0
start_time = time.time()

# =========================================================
# PROCESSING MODULES (TEAMMATES REPLACE THESE)
# =========================================================

def preprocess(audio_frame):
    """
    Basic preprocessing: #MEI#
    - normalization
    - filtering
    - windowing
    """
    return audio_frame


def estimate_direction(audio_frame):
    """
    Direction-of-arrival estimation.
    
    Teammate inserts beamforming /
    cross-correlation code here.
    """
    return 0


def analyze_noise(audio_frame, direction):
    """
    FFT / spectral analysis stage. #MEI#
    """
    return audio_frame


def generate_anti_noise(audio_frame, noise_profile):
    """
    ANC adaptive filter stage, machine learning, etc.
    """
    return 


def process_audio(audio_frame): # Main processing pipeline

    # Step 1
    cleaned = preprocess(audio_frame)

    # Step 2
    direction = estimate_direction(cleaned)

    # Step 3
    noise_profile = analyze_noise(cleaned, direction)

    # Step 4
    anti_noise = generate_anti_noise(cleaned, noise_profile)
    
    
    return anti_noise

# =========================================================
# REAL-TIME AUDIO CALLBACK
# =========================================================

def audio_callback(indata, outdata, frames, time_info, status):

    global callback_count

    # -----------------------------------------------------
    # STATUS / ERROR REPORTING
    # -----------------------------------------------------

    if status:
        print(status)

    # -----------------------------------------------------
    # SAFETY CHECKS
    # -----------------------------------------------------

    if indata.shape[0] != BLOCK_SIZE:
        print("Unexpected block size")

    # -----------------------------------------------------
    # PROCESS AUDIO
    # -----------------------------------------------------

    processed = process_audio(indata)

    # -----------------------------------------------------
    # OUTPUT TO SPEAKERS
    # -----------------------------------------------------

    outdata[:] = processed[:, :OUTPUT_CHANNELS]

    # -----------------------------------------------------
    # PERFORMANCE MONITORING
    # -----------------------------------------------------

    callback_count += 1

# =========================================================
# STREAM SETUP
# =========================================================

stream = sd.Stream(
    samplerate=SAMPLE_RATE,
    blocksize=BLOCK_SIZE,
    channels=(INPUT_CHANNELS, OUTPUT_CHANNELS),
    dtype=DTYPE,
    callback=audio_callback
)

# =========================================================
# START STREAM
# =========================================================

print("Starting ANC stream...")

with stream:
    input("Press Enter to stop...\n")