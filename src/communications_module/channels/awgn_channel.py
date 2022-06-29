import numpy as np
from src.utilities import get_snr_context_rescale_factor


class AWGNChannel:

    def __init__(self,
                 x_in: np.array,
                 rx_snr):

        self.x_in = x_in
        self.rx_snr = rx_snr
        self.noise = self._generate_awgn_noise()

    def _generate_awgn_noise(self):
        if np.isrealobj(self.x_in):
            n = (np.random.standard_normal(self.x_in.shape))
        else:
            n = (np.random.standard_normal(self.x_in.shape) + 1j * np.random.standard_normal(self.x_in.shape))
        return n

    def get_rescale_factor_noise_pow_req_sgn_pow(self):
        factor, noise_power, req_sgn_power = get_snr_context_rescale_factor(self.x_in, self.noise, self.rx_snr)
        return factor, noise_power, req_sgn_power

    def filter_x_in(self):
        factor, noise_power, _ = self.get_rescale_factor_noise_pow_req_sgn_pow()
        y_out = factor * self.x_in
        signal_power = float(np.mean(abs(y_out) ** 2))
        if factor == 1:
            print("RX Signal power: %.4f. Noise power: %.4f, SNR [dB]: %.4f" % (signal_power, noise_power, 0))
            return y_out + self.noise, self.noise
        snr_db = 10 * np.log10(signal_power / noise_power)
        print("RX Signal power: %.4f. Noise power: %.4f, SNR [dB]: %.4f" % (signal_power, noise_power, snr_db))
        return y_out + self.noise, self.noise
