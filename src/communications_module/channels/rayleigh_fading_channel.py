from .abstract_fading_channel import AbstractChannel


class RayleighChannel(AbstractChannel):

    def __init__(self,
                 discrete_path_delays: list,
                 avg_path_gains: list,
                 max_doppler_shift: float):

        super().__init__(discrete_path_delays, avg_path_gains, max_doppler_shift)

    def nth_path_complex_coeffs(self, z_coeffs, path_idx):
        return z_coeffs
