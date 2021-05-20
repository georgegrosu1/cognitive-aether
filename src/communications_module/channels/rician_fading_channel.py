import numpy as np
from .abstract_fading_channel import AbstractChannel


class RicianChannel(AbstractChannel):

    def __init__(self,
                 discrete_path_delays: list,
                 avg_path_gains: list,
                 max_doppler_shift: int,
                 sample_rate: int,
                 k_factors: list,
                 los_doppler_shifts: list,
                 los_init_phases: list):

        super().__init__(discrete_path_delays, avg_path_gains, max_doppler_shift, sample_rate)
        self.k_factors = k_factors
        self.los_dopple_shifts = los_doppler_shifts
        self.los_init_phases = los_init_phases

    def nth_path_complex_coeffs(self, z_coeffs, path_idx):
        first_var = z_coeffs / (np.sqrt(self.k_factors[path_idx]) + 1)
        second_var = np.sqrt(self.k_factors[path_idx] / self.k_factors[path_idx] + 1)
        complex_coeff = np.exp(1j * 2 * np.pi * self.los_dopple_shifts[path_idx] + self.los_init_phases[path_idx])
        rice_complex_coeffs = first_var + second_var * complex_coeff
        return rice_complex_coeffs
