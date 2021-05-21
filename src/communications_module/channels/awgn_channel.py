import numpy as np
from src.utilities import get_snr_context_rescale_factor


class AWGNChannel:

    def __init__(self,
                 x_in: np.array,
                 rx_snr):

        self.x_in = x_in
        self.rx_snr = rx_snr

    def generate_awgn_noise(self):
        if np.isrealobj(self.x_in):
            n = (np.random.standard_normal(self.x_in.shape))
        else:
            n = (np.random.standard_normal(self.x_in.shape) + 1j * np.random.standard_normal(self.x_in.shape))
        return n

    def filter_x_in(self):
        n = self.generate_awgn_noise()
        factor, noise_power, _ = get_snr_context_rescale_factor(self.x_in, n, self.rx_snr)
        y_out = factor * self.x_in
        signal_power = np.mean(abs(y_out ** 2))
        snr_db = 10 * np.log10(signal_power / noise_power)
        print("RX Signal power: %.4f. Noise power: %.4f, SNR [dB]: %.4f" % (signal_power, noise_power, snr_db))
        return y_out + n, n
