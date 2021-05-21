import numpy as np
from src.utilities import draw_from_distribution


class TDLChannel:

    def __init__(self,
                 x_in: np.array,
                 num_taps: int,
                 dist_mean: float = 0,
                 dist_sigma: float = 1 / np.sqrt(2)):
        """
        This is a complete stochastic method implementation for Rayleigh/Rician frequency selective channel
        :parameter x_in a np.array containing samples of input signal
        :parameter num_taps int specifying number of taps. NOTE! One tap is equivalent for a Flat Fading Channel
        :parameter dist_mean float specifying mean of the distribution from which complex coeffs are sampled
        :parameter dist_sigma float specifying variance of the distribution from which complex coeffs are sampled
        For mean 0 and sigma 1 distribution is Rayleigh. Varying mean will lead to Rician distribution
        """

        self.x_in = x_in
        self.num_taps = num_taps
        self.dist_mean = dist_mean
        self.dist_sigma = dist_sigma
        self.tdl_compelx_coeffs = self._init_tdl_coeffs()

    def _init_tdl_coeffs(self):

        def replace_overunity_coeffs(h_z):
            replace_with = np.array([])
            exceeded_idxs = np.where(abs(h_z) >= 1)[0]
            if len(exceeded_idxs) > 0:
                clean_h = np.delete(h_z, exceeded_idxs)
                while len(replace_with) != len(exceeded_idxs):
                    new_coeff = 1 / np.sqrt(2) * draw_from_distribution(samples=1,
                                                                        mean=self.dist_mean,
                                                                        sigma=self.dist_sigma)
                    if abs(new_coeff) < 1:
                        replace_with = np.append(replace_with, new_coeff, axis=0)
                h_z = np.append(clean_h, replace_with, axis=0)
            return h_z

        def get_normalized_ch(taps):
            return 1 / np.sqrt(self.num_taps) * draw_from_distribution(samples=taps, mean=self.dist_mean, sigma=self.dist_sigma)

        h = get_normalized_ch(self.num_taps)
        cleaned_h = replace_overunity_coeffs(h)

        return cleaned_h

    def filter_x_in(self):
        y_out = np.convolve(self.x_in, self.tdl_compelx_coeffs, mode='same')

        return y_out
