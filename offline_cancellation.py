import numpy as np


class OfflineAdaptiveFilter:
    """
    Offline LMS adaptive filter.

    This learns a mapping from a reference signal to a target signal.

    Example use:
        reference_signal = outside mic recording
        target_signal = inside mic recording

    The filter tries to produce:
        predicted_signal ≈ target_signal
    """

    def __init__(self, filter_length=128, learning_rate=0.0001):
        self.filter_length = filter_length
        self.learning_rate = learning_rate

        # Adaptive FIR filter coefficients
        self.weights = np.zeros(filter_length, dtype=np.float32)

        # Stores current and previous reference samples
        self.reference_history = np.zeros(filter_length, dtype=np.float32)

    def reset(self):
        """
        Reset filter memory and learned weights.
        """
        self.weights[:] = 0
        self.reference_history[:] = 0

    def process_sample(self, reference_sample, target_sample):
        """
        Process one sample during offline learning.

        Parameters
        ----------
        reference_sample : float
            Current sample from the reference signal.

        target_sample : float
            Current sample from the target signal.

        Returns
        -------
        predicted_sample : float
            Filter output.

        error : float
            Difference between target and prediction.
        """

        # Shift old samples back
        self.reference_history[1:] = self.reference_history[:-1]

        # Put newest reference sample at the front
        self.reference_history[0] = reference_sample

        # FIR filter output
        predicted_sample = np.dot(self.weights, self.reference_history)

        # Error: what we wanted minus what we predicted
        error = target_sample - predicted_sample

        # LMS update
        self.weights += self.learning_rate * error * self.reference_history

        return predicted_sample, error

    def process(self, reference_signal, target_signal):
        """
        Process full recorded signals offline.

        Parameters
        ----------
        reference_signal : ndarray
            Recorded reference signal.

        target_signal : ndarray
            Recorded target/error microphone signal.

        Returns
        -------
        predicted_signal : ndarray
            The learned estimate of the target signal.

        error_signal : ndarray
            target_signal - predicted_signal
        """

        reference_signal = np.asarray(reference_signal, dtype=np.float32)
        target_signal = np.asarray(target_signal, dtype=np.float32)

        if len(reference_signal) != len(target_signal):
            raise ValueError("reference_signal and target_signal must have the same length.")

        predicted_signal = np.zeros_like(reference_signal)
        error_signal = np.zeros_like(reference_signal)

        for n in range(len(reference_signal)):
            predicted, error = self.process_sample(
                reference_signal[n],
                target_signal[n]
            )

            predicted_signal[n] = predicted
            error_signal[n] = error

        return predicted_signal, error_signal