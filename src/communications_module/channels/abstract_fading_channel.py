from abc import ABC


class AbstractChannel(ABC):

    def __init__(self,
                 discrete_path_delays: list,
                 avg_path_gains: list,
                 max_doppler_shift: int):

        super(AbstractChannel, self).__init__()
        self.discrete_path_delays = discrete_path_delays
        self.avg_path_gains = avg_path_gains
        self.max_doppler_shift = max_doppler_shift
        assert len(discrete_path_delays) == len(avg_path_gains), 'Array length of delays and gains must match'
        self.num_paths = len(avg_path_gains)
        self.ch_response = None

    def nth_path_complex_coeffs(self, z_coeffs, path_idx):
        raise NotImplementedError
