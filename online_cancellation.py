import numpy as np


class OnlineAdaptiveANC:
    """
    Online/live LMS-style adaptive ANC.

    This version is intended for block-by-block real-time processing.

    It uses:
        reference mic = hears incoming noise
        error mic     = hears residual noise near the cancellation point

    It outputs:
        anti_noise signal
    """

    def __init__(
        self,
        filter_length=128,
        learning_rate=0.00001,
        output_gain=0.2
    ):
        self.filter_length = filter_length
        self.learning_rate = learning_rate
        self.output_gain = output_gain

        # Adaptive FIR filter coefficients
        self.weights = np.zeros(filter_length, dtype=np.float32)

        # Stores recent reference mic samples
        self.reference_history = np.zeros(filter_length, dtype=np.float32)

    def reset(self):
        """
        Reset adaptive filter state.
        """
        self.weights[:] = 0
        self.reference_history[:] = 0

    def process_sample(self, reference_sample, error_sample):
        """
        Process one live sample.

        Parameters
        ----------
        reference_sample : float
            Current sample from reference microphone.

        error_sample : float
            Current sample from error microphone.

        Returns
        -------
        anti_noise_sample : float
            Output sample to send to the speaker.
        """

        # Update reference history
        self.reference_history[1:] = self.reference_history[:-1]
        self.reference_history[0] = reference_sample

        # Adaptive filter output
        y = np.dot(self.weights, self.reference_history)

        # Anti-noise output.
        # The minus sign is because we want opposite phase.
        anti_noise_sample = -self.output_gain * y

        # Measured residual error from the error microphone
        e = error_sample

        # LMS-style adaptive update.
        # Depending on hardware polarity, this sign may need flipping.
        self.weights += self.learning_rate * e * self.reference_history

        return anti_noise_sample

    def process_block(self, reference_block, error_block):
        """
        Process one audio block.

        Parameters
        ----------
        reference_block : ndarray
            Block of samples from reference microphone.

        error_block : ndarray
            Block of samples from error microphone.

        Returns
        -------
        anti_noise_block : ndarray
            Block of anti-noise samples to play through speaker.
        """

        reference_block = np.asarray(reference_block, dtype=np.float32)
        error_block = np.asarray(error_block, dtype=np.float32)

        if len(reference_block) != len(error_block):
            raise ValueError("reference_block and error_block must have the same length.")

        anti_noise_block = np.zeros_like(reference_block)

        for n in range(len(reference_block)):
            anti_noise_block[n] = self.process_sample(
                reference_block[n],
                error_block[n]
            )

        # Prevent speaker-damaging / clipping output
        anti_noise_block = np.clip(anti_noise_block, -1.0, 1.0)

        return anti_noise_block