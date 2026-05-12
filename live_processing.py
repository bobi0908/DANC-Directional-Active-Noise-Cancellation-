# region
import numpy as np
import sounddevice as sd

# Audio settings
SAMPLE_RATE = 44100
DURATION = 1.0  # seconds

# Musical note names
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
              'F#', 'G', 'G#', 'A', 'A#', 'B']


def frequency_to_note(freq):
    """
    Convert frequency to nearest musical note.
    A4 = 440 Hz
    """
    if freq <= 0:
        return None

    # MIDI note number
    midi = int(round(69 + 12 * np.log2(freq / 440.0)))

    note_name = NOTE_NAMES[midi % 12]
    octave = (midi // 12) - 1

    return f"{note_name}{octave}"


def detect_frequency(audio, sample_rate):
    """
    Detect dominant frequency using FFT.
    """
    # Apply window to reduce noise
    windowed = audio * np.hanning(len(audio))

    # FFT
    fft = np.fft.rfft(windowed)
    frequencies = np.fft.rfftfreq(len(windowed), 1 / sample_rate)

    # Magnitude
    magnitude = np.abs(fft)

    # Ignore very low frequencies
    min_freq = 50
    max_freq = 2000

    valid = np.where((frequencies >= min_freq) &
                     (frequencies <= max_freq))

    frequencies = frequencies[valid]
    magnitude = magnitude[valid]

    peak_index = np.argmax(magnitude)
    dominant_freq = frequencies[peak_index]

    return dominant_freq


print("Listening... Press Ctrl+C to stop.")

try:
    while True:
        # Record audio
        audio = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float64'
        )

        sd.wait()

        # Flatten audio array
        audio = audio.flatten()

        # Detect frequency
        freq = detect_frequency(audio, SAMPLE_RATE)

        # Convert to note
        note = frequency_to_note(freq)

        print(f"Frequency: {freq:.2f} Hz  ->  Note: {note}")

except KeyboardInterrupt:
    print("Stopped.")
# endregion

# region 2
import numpy as np
import sounddevice as sd

SAMPLE_RATE = 44100
BLOCK_SIZE = 2048

# Adjustable volume of cancelling tone
CANCEL_VOLUME = 0.3

# Store phase continuity between blocks
phase = 0


def detect_frequency(audio, sample_rate):
    """
    Detect dominant frequency using FFT.
    """
    audio = audio.flatten()

    # Apply window
    windowed = audio * np.hanning(len(audio))

    # FFT
    fft = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(len(windowed), 1 / sample_rate)

    magnitudes = np.abs(fft)

    # Ignore low frequencies
    valid = np.where((freqs > 50) & (freqs < 2000))

    freqs = freqs[valid]
    magnitudes = magnitudes[valid]

    if len(magnitudes) == 0:
        return 0

    peak = np.argmax(magnitudes)

    return freqs[peak]


def callback(indata, outdata, frames, time, status):
    global phase

    if status:
        print(status)

    # Detect dominant frequency
    freq = detect_frequency(indata, SAMPLE_RATE)

    # Time array
    t = np.arange(frames) / SAMPLE_RATE

    # Generate opposite-phase sine wave
    sinewave = CANCEL_VOLUME * np.sin(
        2 * np.pi * freq * t + phase + np.pi
    )

    # Keep phase continuous
    phase += 2 * np.pi * freq * frames / SAMPLE_RATE
    phase = np.mod(phase, 2 * np.pi)

    # Output as mono signal
    outdata[:] = sinewave.reshape(-1, 1)

    print(f"Detected frequency: {freq:.1f} Hz", end="\r")


print("Tone canceller running...")
print("Use headphones to avoid feedback.")
print("Press Ctrl+C to stop.")

try:
    with sd.Stream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=1,
        dtype='float32',
        callback=callback
    ):
        while True:
            sd.sleep(100)

except KeyboardInterrupt:
    print("\nStopped.")
# endregion

