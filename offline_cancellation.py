import numpy as np


class OfflineAdaptiveFilter:
   

    def __init__(self, filter_length=128, learning_rate=0.0001):
        self.filter_length = filter_length
        self.learning_rate = learning_rate

        # Adaptive FIR filter coefficients
        self.weights = np.zeros(filter_length, dtype=np.float32)

        # Stores current and previous reference samples
        self.reference_history = np.zeros(filter_length, dtype=np.float32)

    def reset(self):
       
        self.weights[:] = 0
        self.reference_history[:] = 0

    def process_sample(self, reference_sample, target_sample):
       

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