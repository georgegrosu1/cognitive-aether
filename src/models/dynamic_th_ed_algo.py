import numpy as np
from src.utilities import invqfunc


class EDDector:

    def __init__(self, window_size):
        self.sensing_window_size = window_size

    def predict(self, pu_signal_vector, awgn_noise_vector, fals_proba):
        sigma = np.var(awgn_noise_vector)
        energy = np.mean(abs(pu_signal_vector)**2)
        thresh = sigma * (invqfunc(fals_proba) * np.sqrt(2 * self.sensing_window_size) + self.sensing_window_size)
        if energy >= thresh:
            return 1
        return 0
