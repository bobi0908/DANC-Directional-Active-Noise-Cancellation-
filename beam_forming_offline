#channel audio input
#split into mic1 and mic2
#compare amplitudes
#estimate sample delay
#¢return useful analysis

#essentially we are trying to use iphone/ anhy working microphone 
# set up and then develope the math/coding
#for the beamforming and noise cancellation in the offline setting first and then 
#once we know that works we can replace the microphones to the actual mics/raspberry pi and then test in the real world
import os
import shutil
import subprocess
import tempfile

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import resample_poly


def split_two_channels(audio_frame):
    """
    Split stereo/2-channel audio into mic1 and mic2.
    Expected shape: (samples, 2)
    """

    if audio_frame.ndim != 2:
        raise ValueError("audio_frame must be 2D with shape (samples, channels).")

    if audio_frame.shape[1] < 2:
        raise ValueError("Need at least 2 microphone channels.")

    mic1 = audio_frame[:, 0]
    mic2 = audio_frame[:, 1]

    return mic1, mic2


def compare_microphones(mic1, mic2):
    """
    Extract basic comparison features between two microphones.
    """

    features = {
        "mic1_max": np.max(np.abs(mic1)),
        "mic2_max": np.max(np.abs(mic2)),
        "mic1_mean": np.mean(mic1),
        "mic2_mean": np.mean(mic2),
        "mic1_rms": np.sqrt(np.mean(mic1 ** 2)),
        "mic2_rms": np.sqrt(np.mean(mic2 ** 2)),
    }

    return features


def estimate_delay_samples(mic1, mic2):
    """
    Estimate delay between mic1 and mic2 using cross-correlation.

    Positive delay means mic2 is delayed relative to mic1.
    Negative delay means mic1 is delayed relative to mic2.
    """

    mic1 = mic1 - np.mean(mic1)
    mic2 = mic2 - np.mean(mic2)

    correlation = np.correlate(mic2, mic1, mode="full")

    delay_index = np.argmax(correlation)
    delay_samples = delay_index - (len(mic1) - 1)

    return delay_samples


def analyse_two_mics(audio_frame, sample_rate=48000):
    """
    Full first-stage beamforming analysis.
    """

    mic1, mic2 = split_two_channels(audio_frame)

    features = compare_microphones(mic1, mic2)

    delay_samples = estimate_delay_samples(mic1, mic2)
    delay_seconds = delay_samples / sample_rate

    analysis = {
        "mic1": mic1,
        "mic2": mic2,
        "features": features,
        "delay_samples": delay_samples,
        "delay_seconds": delay_seconds,
    }

    return analysis


# =============================================================================
# GEOMETRY CONFIGURATION — SET YOUR PHYSICAL DIMENSIONS HERE
# =============================================================================
# All positions are in metres (x, y, z)
#
# Coordinate system (define once, use everywhere):
#
#        +Y (forward)
#         |
#         |
#         |_______ +X (right)
#        /
#       /
#     +Z (up)
#
# Example layout (top-down view, mics and speaker on same horizontal plane):
#
#        MIC1 ----d---- MIC2
#                 |
#               (gap)
#                 |
#              SPEAKER
#
# Set Z = 0 for all if everything is on the same flat plane/table.
# =============================================================================
#CHANGE THIS SECTION DUE TO DIMENSIONS 
GEOMETRY = {
    # --- Microphone positions (metres) ---
    # Place mic1 at the origin, mic2 along the X axis
    "mic1_pos"      : np.array([0.00, 0.00, 0.00]),   # mic1 at origin
    "mic2_pos"      : np.array([0.05, 0.00, 0.00]),   # mic2 5cm to the right

    # --- Speaker position (metres) ---
    # Speaker is in front of the mic array along Y axis
    # Change this to match where your speaker actually is
    "speaker_pos"   : np.array([0.025, 0.20, 0.00]),  # 20cm in front, centred

    # --- Reference direction (unit vector pointing "forward") ---
    # This defines what 0 degrees means
    # Default: forward is along +Y axis
    "forward_dir"   : np.array([0.00, 1.00, 0.00]),

    # --- Mic axis (unit vector from mic1 to mic2) ---
    # Computed automatically below — do not edit this line
    "mic_axis"      : None,
}

# Auto-compute mic axis from positions
_mic_vec = GEOMETRY["mic2_pos"] - GEOMETRY["mic1_pos"]
GEOMETRY["mic_axis"] = _mic_vec / np.linalg.norm(_mic_vec)

# Auto-compute mic spacing from positions
GEOMETRY["mic_spacing_m"] = float(np.linalg.norm(_mic_vec))


def describe_geometry(geometry):
    """
    Print a human-readable summary of the physical setup.
    Useful to sanity check your dimensions before running.
    """
    spacing = geometry["mic_spacing_m"]
    spk     = geometry["speaker_pos"]
    m1      = geometry["mic1_pos"]
    m2      = geometry["mic2_pos"]
    midpoint = (m1 + m2) / 2
    spk_dist = float(np.linalg.norm(spk - midpoint))

    print("=" * 55)
    print("         PHYSICAL GEOMETRY SETUP")
    print("=" * 55)
    print(f"  Mic 1 position   : {m1} m")
    print(f"  Mic 2 position   : {m2} m")
    print(f"  Mic spacing      : {spacing * 100:.1f} cm")
    print(f"  Mic axis         : {geometry['mic_axis']}")
    print(f"  Speaker position : {spk} m")
    print(f"  Speaker distance : {spk_dist * 100:.1f} cm from mic midpoint")
    print(f"  Forward direction: {geometry['forward_dir']}")
    print("=" * 55)


# =============================================================================
# STAGE 1: CHANNEL SPLITTING & BASIC FEATURES
# =============================================================================

def split_two_channels(audio_frame):
    """
    Split stereo/2-channel audio into mic1 and mic2.

    Parameters:
        audio_frame : 2D numpy array of shape (samples, 2)

    Returns:
        mic1 : 1D numpy array — first microphone signal
        mic2 : 1D numpy array — second microphone signal
    """
    if audio_frame.ndim != 2:
        raise ValueError("audio_frame must be 2D with shape (samples, channels).")
    if audio_frame.shape[1] < 2:
        raise ValueError("Need at least 2 microphone channels.")

    mic1 = audio_frame[:, 0]
    mic2 = audio_frame[:, 1]

    return mic1, mic2


def compare_microphones(mic1, mic2):
    """
    Extract basic comparison features between two microphones.

    Parameters:
        mic1 : 1D numpy array — first microphone signal
        mic2 : 1D numpy array — second microphone signal

    Returns:
        features : dict of amplitude and energy features
    """
    features = {
        "mic1_max"  : np.max(np.abs(mic1)),
        "mic2_max"  : np.max(np.abs(mic2)),
        "mic1_mean" : np.mean(mic1),
        "mic2_mean" : np.mean(mic2),
        "mic1_rms"  : np.sqrt(np.mean(mic1 ** 2)),
        "mic2_rms"  : np.sqrt(np.mean(mic2 ** 2)),
    }
    return features


# =============================================================================
# STAGE 2: DELAY ESTIMATION
# =============================================================================

def estimate_delay_samples(mic1, mic2):
    """
    Estimate delay between mic1 and mic2 using cross-correlation.

    Positive delay = mic2 is delayed relative to mic1
                     meaning sound came from the mic1 side.
    Negative delay = mic1 is delayed relative to mic2
                     meaning sound came from the mic2 side.

    Parameters:
        mic1 : 1D numpy array — first microphone signal
        mic2 : 1D numpy array — second microphone signal

    Returns:
        delay_samples : integer sample delay (mic2 relative to mic1)
    """
    mic1 = mic1 - np.mean(mic1)
    mic2 = mic2 - np.mean(mic2)

    correlation   = np.correlate(mic2, mic1, mode="full")
    delay_index   = np.argmax(correlation)
    delay_samples = delay_index - (len(mic1) - 1)

    return delay_samples


def analyse_two_mics(audio_frame, sample_rate=48000):
    """
    Full first-stage beamforming analysis.

    Parameters:
        audio_frame : 2D numpy array of shape (samples, 2)
        sample_rate : integer sample rate in Hz (default 48000)

    Returns:
        analysis : dict containing mic signals, features, and delay info
    """
    mic1, mic2    = split_two_channels(audio_frame)
    features      = compare_microphones(mic1, mic2)
    delay_samples = estimate_delay_samples(mic1, mic2)
    delay_seconds = delay_samples / sample_rate

    analysis = {
        "mic1"          : mic1,
        "mic2"          : mic2,
        "features"      : features,
        "delay_samples" : delay_samples,
        "delay_seconds" : delay_seconds,
    }
    return analysis


# =============================================================================
# STAGE 3: DIRECTION OF ARRIVAL (DoA) — WITH GEOMETRY
# =============================================================================

def estimate_doa(delay_seconds, geometry, speed_of_sound=343.0):
    """
    Estimate Direction of Arrival using the physical geometry.

    The angle is measured relative to the forward_dir defined in GEOMETRY.
    Because we only have 2 mics, we can only resolve the angle along
    the mic axis — we cannot resolve the full 3D cone of confusion
    without more microphones.

    Parameters:
        delay_seconds  : float  — time delay between mic1 and mic2 in seconds
        geometry       : dict   — the GEOMETRY config dict
        speed_of_sound : float  — speed of sound in m/s

    Returns:
        doa : dict with angle and direction vector info
    """
    mic_spacing = geometry["mic_spacing_m"]

    # Angle along the mic axis
    sin_theta = np.clip(
        (delay_seconds * speed_of_sound) / mic_spacing,
        -1.0,
        1.0
    )
    angle_rad = np.arcsin(sin_theta)
    angle_deg = np.degrees(angle_rad)

    # Compute angle relative to the defined forward direction
    # Project mic_axis onto forward_dir to get the offset
    mic_axis    = geometry["mic_axis"]
    forward_dir = geometry["forward_dir"]
    axis_angle_from_forward = np.degrees(
        np.arccos(np.clip(np.dot(mic_axis, forward_dir), -1.0, 1.0))
    )

    # Absolute angle from forward direction
    absolute_angle_deg = angle_deg + (90.0 - axis_angle_from_forward)

    return {
        "angle_along_mic_axis_deg" : angle_deg,
        "absolute_angle_deg"       : absolute_angle_deg,
        "mic_spacing_used_m"       : mic_spacing,
        "note": (
            "Only 1D angle resolved along mic axis. "
            "Full 3D localisation requires more microphones."
        )
    }


# =============================================================================
# STAGE 4: DELAY-AND-SUM BEAMFORMER
# =============================================================================

def apply_integer_delay(signal, delay_samples):
    """
    Apply an integer sample delay to a signal using np.roll,
    then zero out the wrapped samples to avoid circular artefacts.

    Parameters:
        signal        : 1D numpy array
        delay_samples : integer number of samples to shift

    Returns:
        shifted : delayed signal of the same length
    """
    int_delay = int(np.round(delay_samples))
    shifted   = np.roll(signal, int_delay)

    if int_delay > 0:
        shifted[:int_delay] = 0
    elif int_delay < 0:
        shifted[int_delay:] = 0

    return shifted


def delay_and_sum_beamformer(mic1, mic2, delay_samples, w1=0.5, w2=0.5):
    """
    Delay-and-Sum beamformer for 2 microphones.

    Parameters:
        mic1, mic2     : 1D numpy arrays — microphone signals
        delay_samples  : integer — estimated delay (mic2 relative to mic1)
        w1, w2         : float — beamforming weights (default equal weighting)

    Returns:
        beamformed_output : 1D numpy array — aligned and summed signal
    """
    mic2_aligned      = apply_integer_delay(mic2, -delay_samples)
    beamformed_output = w1 * mic1 + w2 * mic2_aligned

    return beamformed_output


# =============================================================================
# STAGE 5: FULL PHASE 1 PIPELINE
# =============================================================================

def phase1_pipeline(audio_frame, geometry, sample_rate=48000):
    """
    Full offline Phase 1 beamforming validation pipeline.

    Parameters:
        audio_frame : 2D numpy array of shape (samples, 2)
        geometry    : dict — the GEOMETRY config dict defined at the top
        sample_rate : int  — sample rate in Hz (default 48000)

    Returns:
        results : dict containing all intermediate and final outputs
    """
    # Stage 1 & 2 — split, features, delay
    analysis = analyse_two_mics(audio_frame, sample_rate)

    # Stage 3 — Direction of Arrival using geometry
    doa = estimate_doa(analysis["delay_seconds"], geometry)

    # Stage 4 — Beamforming
    beamformed = delay_and_sum_beamformer(
        analysis["mic1"],
        analysis["mic2"],
        analysis["delay_samples"]
    )

    return {
        **analysis,
        "doa"      : doa,
        "geometry" : geometry,
    }


# =============================================================================
# STAGE 6: PRE-RECORDED AUDIO LOADER
# =============================================================================


def _to_mono(signal):
    """
    Convert a WAV signal to mono by taking the first channel if needed.
    """
    arr = np.asarray(signal)
    if arr.ndim == 2:
        arr = arr[:, 0]
    return arr.astype(np.float32)


def detect_alsa_capture_devices():
    """
    Use `arecord -l` to discover available Raspberry Pi ALSA capture devices.
    Returns a list like ['hw:1,0', 'hw:2,0'] when available.
    """
    if shutil.which("arecord") is None:
        return []

    try:
        result = subprocess.run(
            ["arecord", "-l"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    devices = []
    for line in result.stdout.splitlines():
        if "card" in line and "device" in line:
            parts = line.split(':', 2)
            if len(parts) >= 2:
                card = parts[1].strip()
                # keep the raw device name for ALSA use
                devices.append(f"hw:{card.split()[0]},0")
    return list(dict.fromkeys(devices))


def record_alsa_stereo_device(
    device_name,
    output_path,
    sample_rate=48000,
    duration=3.0,
    channels=2,
    formats=("S32_LE", "S16_LE")
):
    """
    Record one ALSA capture device to a 2-channel WAV file using `arecord`.
    This is the Raspberry Pi / I2S-friendly path for offline analysis.
    """
    if shutil.which("arecord") is None:
        raise FileNotFoundError("arecord not found. Install ALSA tools on the Pi first.")

    for fmt in formats:
        try:
            subprocess.run([
                "arecord",
                "-D", device_name,
                "-c", str(channels),
                "-r", str(sample_rate),
                "-f", fmt,
                "-d", str(int(duration)),
                "-t", "wav",
                "-q",
                output_path,
            ], check=True)
            return fmt
        except subprocess.CalledProcessError:
            continue

    raise RuntimeError(
        f"Failed to record stereo audio from {device_name}. "
        f"Tried formats: {', '.join(formats)}"
    )


def _to_stereo(signal):
    """
    Normalize WAV data to a 2-channel float32 array.
    """
    arr = np.asarray(signal)
    if arr.ndim == 1:
        raise ValueError("Expected stereo WAV file with 2 channels.")
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, :2].astype(np.float32)
    raise ValueError("Unsupported WAV shape for stereo audio: {}".format(arr.shape))


def load_stereo_audio_file(stereo_path, target_sample_rate=48000):
    """
    Load a prerecorded 2-channel WAV file for the offline beamforming pipeline.

    Parameters:
        stereo_path        : str — path to stereo WAV file
        target_sample_rate : int — common sample rate to resample to

    Returns:
        audio_frame : 2D numpy array of shape (samples, 2)
    """
    if not os.path.exists(stereo_path):
        raise FileNotFoundError(f"Stereo file not found: {stereo_path}")

    sr, stereo = wavfile.read(stereo_path)
    stereo = _to_stereo(stereo)

    if sr != target_sample_rate:
        left = resample_poly(stereo[:, 0], target_sample_rate, sr).astype(np.float32)
        right = resample_poly(stereo[:, 1], target_sample_rate, sr).astype(np.float32)
        stereo = np.stack([left, right], axis=1)

    peak = np.max(np.abs(stereo)) if stereo.size else 1.0
    stereo = stereo / max(peak, 1e-8)

    return stereo


def load_two_audio_files(mic1_path, mic2_path, target_sample_rate=48000):
    """
    Load two prerecorded mono WAV files and return them as a stereo-style
    2D array of shape (samples, 2) for the offline beamforming pipeline.

    Parameters:
        mic1_path          : str — path to first microphone recording
        mic2_path          : str — path to second microphone recording
        target_sample_rate : int — common sample rate to resample to

    Returns:
        audio_frame : 2D numpy array of shape (samples, 2)
    """
    if not os.path.exists(mic1_path):
        raise FileNotFoundError(f"Microphone file not found: {mic1_path}")
    if not os.path.exists(mic2_path):
        raise FileNotFoundError(f"Microphone file not found: {mic2_path}")

    sr1, mic1 = wavfile.read(mic1_path)
    sr2, mic2 = wavfile.read(mic2_path)

    mic1 = _to_mono(mic1)
    mic2 = _to_mono(mic2)

    if sr1 != sr2:
        common_sr = max(sr1, sr2)
        if sr1 != common_sr:
            mic1 = resample_poly(mic1, common_sr, sr1).astype(np.float32)
        if sr2 != common_sr:
            mic2 = resample_poly(mic2, common_sr, sr2).astype(np.float32)
    else:
        common_sr = sr1

    if common_sr != target_sample_rate:
        mic1 = resample_poly(mic1, target_sample_rate, common_sr).astype(np.float32)
        mic2 = resample_poly(mic2, target_sample_rate, common_sr).astype(np.float32)

    length = min(len(mic1), len(mic2))
    mic1 = mic1[:length]
    mic2 = mic2[:length]

    peak1 = np.max(np.abs(mic1)) if len(mic1) else 1.0
    peak2 = np.max(np.abs(mic2)) if len(mic2) else 1.0
    scale = max(peak1, peak2, 1e-8)
    mic1 = mic1 / scale
    mic2 = mic2 / scale

    return np.stack([mic1, mic2], axis=1)


# =============================================================================
# STAGE 7: SYNTHETIC TEST SIGNAL GENERATOR
# =============================================================================

def generate_test_signal(
    sample_rate=48000,
    duration=0.5,
    freq=440,
    delay_samples=10,
    noise_level=0.01
):
    """
    Generate a synthetic stereo test signal with a known inter-channel delay.

    Parameters:
        sample_rate   : int   — sample rate in Hz
        duration      : float — signal duration in seconds
        freq          : float — sine wave frequency in Hz
        delay_samples : int   — number of samples to delay mic2
        noise_level   : float — amplitude of added white noise

    Returns:
        audio_frame : 2D numpy array of shape (samples, 2)
    """
    t      = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = np.sin(2 * np.pi * freq * t)

    mic1 = signal.copy()
    mic2 = np.roll(signal, delay_samples)
    mic2[:delay_samples] = 0

    rng   = np.random.default_rng(seed=42)
    mic1 += rng.normal(0, noise_level, size=mic1.shape)
    mic2 += rng.normal(0, noise_level, size=mic2.shape)

    return np.stack([mic1, mic2], axis=1)


# =============================================================================
# STAGE 7: VISUALISATION
# =============================================================================

def plot_results(results, sample_rate=48000):
    """
    Plot Phase 1 pipeline outputs for offline validation.

    Panels:
        1. Raw mic signals
        2. Aligned mic signals after delay compensation
        3. Beamformed output

    Parameters:
        results     : dict — output from phase1_pipeline()
        sample_rate : int  — sample rate in Hz
    """
    t   = np.arange(len(results["mic1"])) / sample_rate
    doa = results["doa"]

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    fig.suptitle("Directional ANC — Direction Estimation Only", fontsize=14, fontweight="bold")

    # --- Panel 1: Raw mic signals ---
    axes[0].plot(t, results["mic1"], label="Mic 1", alpha=0.85, color="steelblue")
    axes[0].plot(t, results["mic2"], label="Mic 2", alpha=0.85, color="darkorange", linestyle="--")
    axes[0].set_title(
        f'Raw Mic Signals  |  Estimated Delay: {results["delay_samples"]} samples '
        f'({results["delay_seconds"] * 1000:.3f} ms)'
    )
    axes[0].legend(loc="upper right")

    # --- Panel 2: Aligned signals ---
    mic2_aligned = apply_integer_delay(results["mic2"], -results["delay_samples"])
    axes[1].plot(t, results["mic1"], label="Mic 1 (reference)", alpha=0.85, color="steelblue")
    axes[1].plot(t, mic2_aligned,    label="Mic 2 (aligned)",   alpha=0.85, color="green", linestyle="--")
    axes[1].set_title("Aligned Signals — Mic 2 delay compensated (should overlap with Mic 1)")
    axes[1].legend(loc="upper right")

    # --- Panel 3 removed: this version now estimates direction only, without anti-noise output. ---

    for ax in axes:
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([t[0], t[-1]])

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


def print_summary(results, known_delay=None):
    """
    Print a formatted summary of the Phase 1 pipeline results.

    Parameters:
        results     : dict — output from phase1_pipeline()
        known_delay : int or None — true delay used in test signal
    """
    doa = results["doa"]
    print("=" * 55)
    print("     DIRECTIONAL ANC — PHASE 1 SUMMARY")
    print("=" * 55)
    print(f"  Estimated delay        : {results['delay_samples']} samples "
          f"({results['delay_seconds'] * 1000:.4f} ms)")
    if known_delay is not None:
        print(f"  Known true delay       : {known_delay} samples")
        print(f"  Estimation error       : {abs(results['delay_samples'] - known_delay)} sample(s)")
    print(f"  Angle along mic axis   : {doa['angle_along_mic_axis_deg']:.2f} degrees")
    print(f"  Absolute angle         : {doa['absolute_angle_deg']:.2f} degrees")
    print(f"  Mic spacing used       : {doa['mic_spacing_used_m'] * 100:.1f} cm")
    print(f"  Note                   : {doa['note']}")
    print("-" * 55)
    print("  Microphone Features:")
    for key, val in results["features"].items():
        print(f"    {key:<12} : {val:.6f}")
    print("=" * 55)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    # --- Audio configuration ---
    SAMPLE_RATE   = 48000
    DURATION      = 0.5
    FREQ          = 440
    KNOWN_DELAY   = 15
    NOISE_LEVEL   = 0.01

    # Use prerecorded WAV files if you already have them.
    # On Raspberry Pi 5, these can also be generated automatically from ALSA/I2S devices.
    STEREO_FILE    = "mic_stereo.wav"
    MIC1_FILE      = "mic1.wav"
    MIC2_FILE      = "mic2.wav"
    RECORD_SECONDS = 3

    # --- Print geometry so you can sanity check it before running ---
    describe_geometry(GEOMETRY)

    # --- Prefer prerecorded files if they exist. Otherwise try ALSA/I2S capture on Pi. ---
    if os.path.exists(STEREO_FILE):
        print(f"Loading prerecorded stereo file: {STEREO_FILE}")
        audio_frame = load_stereo_audio_file(STEREO_FILE, target_sample_rate=SAMPLE_RATE)
        known_delay = None
    elif os.path.exists(MIC1_FILE) and os.path.exists(MIC2_FILE):
        print(f"Loading prerecorded files: {MIC1_FILE} and {MIC2_FILE}")
        audio_frame = load_two_audio_files(MIC1_FILE, MIC2_FILE, target_sample_rate=SAMPLE_RATE)
        known_delay = None
    else:
        print("No prerecorded WAV files found. Trying Pi ALSA/I2S capture...")
        devices = detect_alsa_capture_devices()
        print("Detected ALSA capture devices:", devices)
        if len(devices) >= 1:
            stereo_path = os.path.abspath(STEREO_FILE)
            recorded_format = record_alsa_stereo_device(
                devices[0],
                stereo_path,
                sample_rate=SAMPLE_RATE,
                duration=RECORD_SECONDS,
            )
            print(f"Recorded stereo WAV using format: {recorded_format}")
            audio_frame = load_stereo_audio_file(stereo_path, target_sample_rate=SAMPLE_RATE)
            known_delay = None
        else:
            print("No ALSA capture devices found. Using synthetic demo signal instead.")
            audio_frame = generate_test_signal(
                sample_rate   = SAMPLE_RATE,
                duration      = DURATION,
                freq          = FREQ,
                delay_samples = KNOWN_DELAY,
                noise_level   = NOISE_LEVEL
            )
            known_delay = KNOWN_DELAY

    # --- Run Phase 1 pipeline ---
    results = phase1_pipeline(
        audio_frame = audio_frame,
        geometry    = GEOMETRY,
        sample_rate = SAMPLE_RATE,
    )

    # --- Print summary ---
    print_summary(results, known_delay=known_delay)

    # --- Plot results ---
    plot_results(results, sample_rate=SAMPLE_RATE)