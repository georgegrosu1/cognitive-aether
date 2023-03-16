import numpy as np
from scipy.stats import truncnorm
from scipy.signal import resample_poly


class RandomSampling:
    def __init__(self,
                 keep_rate: float,
                 decimation: int,
                 sampling_type: str):

        assert ((sampling_type == 'jrs_normal') | (sampling_type == 'jrs_uniform') |
                (sampling_type == 'ars_normal') | (sampling_type == 'ars_uniform')), \
            f'{sampling_type.upper()} not supported'

        assert(0 < keep_rate <= 1), "Keep rate must be of (0, 1]"

        self.keep_rate = keep_rate
        self.decimation = decimation
        self.sampling_type = sampling_type

    def jrs_random_sampling(self, signal_length):
        sampling_idxs = np.array([])
        half_range = self.decimation / 2
        avg_vals = np.arange(half_range, signal_length, self.decimation)
        for avg_val in avg_vals:
            if 'normal' in self.sampling_type:
                selection = truncnorm(a=-half_range, b=(half_range-1), loc=avg_val).rvs(size=(1,))
            else:
                selection = np.random.uniform(low=(avg_val-half_range), high=(avg_val+half_range-1), size=(1,))
            sampling_idxs = np.concatenate([sampling_idxs, selection])
        sampling_idxs = sampling_idxs.round()

        return sampling_idxs.astype(int)

    def ars_random_sampling(self, signal_length, sampling_idxs: np.ndarray = np.array([0])):
        if 'normal' in self.sampling_type:
            next_idx = truncnorm(a=-self.decimation,
                                 b=(self.decimation-1),
                                 loc=(sampling_idxs[-1]+self.decimation)).rvs(size=(1,))

            if next_idx[0] > signal_length - 1:
                sampling_idxs = sampling_idxs.round()
                return sampling_idxs.astype(int)

            sampling_idxs = np.concatenate([sampling_idxs, next_idx])
            return self.ars_random_sampling(signal_length, sampling_idxs)
        else:
            next_idx = np.random.uniform(low=sampling_idxs[-1]+1,
                                         high=(sampling_idxs[-1]+2*self.decimation)-1,
                                         size=(1,))

            if next_idx[0] > signal_length - 1:
                sampling_idxs = sampling_idxs.round()
                return sampling_idxs.astype(int)

            sampling_idxs = np.concatenate([sampling_idxs, next_idx])
            return self.ars_random_sampling(signal_length, sampling_idxs)

    def get_nonuniform_signal(self, signal: np.ndarray):
        req_len = int(signal.shape[0] * self.keep_rate) * self.decimation
        lcm = np.lcm(req_len, signal.shape[0])
        resamp_signal = resample_poly(signal,
                                      lcm//signal.shape[0],
                                      lcm//req_len)
        if 'jrs' in self.sampling_type:
            rand_indexes = self.jrs_random_sampling(resamp_signal.shape[0])
        else:
            rand_indexes = self.ars_random_sampling(resamp_signal.shape[0])
        return resamp_signal[rand_indexes]
