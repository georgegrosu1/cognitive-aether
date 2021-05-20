import numpy as np
from .rician_fading_channel import RicianChannel
from .rayleigh_fading_channel import RayleighChannel


class FadingChannel:

    def __init__(self,
                 x_in: np.array,
                 channel_type: str,
                 discrete_path_delays: list,
                 avg_path_gains: list,
                 max_doppler_shift: int,
                 sample_rate: int = None,
                 num_sinusoids: int = 20,
                 k_factors: list = (),
                 los_doppler_shifts: list = (0,),
                 los_init_phases: list = (0,)):

        def _init_sample_rate():
            if sample_rate is None:
                return 100 * x_in.shape[0]
            return sample_rate

        def _init_channel_model():
            if channel_type == 'rician':
                assert len(k_factors) > 0, 'Provide at least one K factor value'
                return RicianChannel(discrete_path_delays, avg_path_gains, max_doppler_shift, self.sample_rate,
                                     k_factors, los_doppler_shifts, los_init_phases)
            elif channel_type == 'rayleigh':
                return RayleighChannel(discrete_path_delays, avg_path_gains, max_doppler_shift, self.sample_rate)

        self.x_in = x_in
        assert num_sinusoids > 0, 'Number of sinusoids must be positive integer'
        self.sample_rate = _init_sample_rate()
        self.__n1n2_sinusoids = (num_sinusoids, num_sinusoids + 1)
        self.model = _init_channel_model()
        self.gmeds_tap_weights = self._init_gmeds_ch_taps()

    def _init_gmeds_ch_taps(self):
        return self.gmeds_fading_ch()

    def compute_um_i(self, num_samples, re_or_im_idx, multipath_idx, waves_p_path):
        t = np.arange(0, num_samples / self.model.sample_rate, 1 / self.model.sample_rate)
        um = np.zeros(num_samples)
        for n in range(waves_p_path):
            alfa = ((-1) ** (re_or_im_idx - 1)) * (np.pi / (4 * waves_p_path)) * \
                   (multipath_idx / (self.model.num_paths + 2))
            f_kn = self.model.max_doppler_shift * np.cos((np.pi / (2 * waves_p_path)) * (n - 0.5) + alfa)
            um += np.cos(2 * np.pi * f_kn * t + np.random.uniform(0.001, 2 * np.pi))
        return np.sqrt(2 / waves_p_path) * um

    def mutually_uncorr_fading_waveforms(self, num_samples):
        z_multipaths = []
        re_paths, im_paths = [], []

        for nth_multipath in range(self.model.num_paths):
            for re_im_idx in range(1, 3):
                if re_im_idx == 1:
                    um_i = self.compute_um_i(num_samples, re_im_idx,
                                             nth_multipath, self.__n1n2_sinusoids[0])
                    re_paths.append(um_i)
                else:
                    um_i = self.compute_um_i(num_samples, re_im_idx,
                                             nth_multipath, self.__n1n2_sinusoids[1])
                    im_paths.append(um_i)
            z_multipaths.append(np.array(re_paths[nth_multipath] + 1j * im_paths[nth_multipath]))

        return z_multipaths

    def gmeds_algo_complex_coeffs(self, num_samples: int):
        a_k_multipaths = []
        z_multipaths = self.mutually_uncorr_fading_waveforms(num_samples)
        for idx, z_path in enumerate(z_multipaths):
            coeff_idx = idx * (len(self.model.k_factors) > 1)
            z_k = self.model.nth_path_complex_coeffs(z_path, coeff_idx)
            omega = self.rescale_factor_for_avg_gain(self.model.avg_path_gains[idx], z_k)
            a_k = np.sqrt(omega) * z_k
            a_k_multipaths.append(a_k)

        return a_k_multipaths

    def generate_gmeds_tap_weights(self, a_k_paths, sample_period, discrete_delays):
        tap_weights = []
        for n in range(-self.__n1n2_sinusoids[0], self.__n1n2_sinusoids[1] + 1):
            g_n = np.zeros(shape=a_k_paths[0].shape)
            for k in range(len(a_k_paths)):
                g_n += a_k_paths[k] * np.sinc((discrete_delays[k] / sample_period) - n)
            tap_weights.append(g_n)

        return tap_weights

    def gmeds_fading_ch(self):
        num_samples = self.x_in.shape[0]
        a_k_mpaths = self.gmeds_algo_complex_coeffs(num_samples)
        gn_tap_weights = self.generate_gmeds_tap_weights(a_k_mpaths,
                                                         1/self.model.sample_rate,
                                                         self.model.discrete_path_delays)
        return gn_tap_weights

    def filter_x_in(self):
        y_out = np.zeros(shape=self.x_in.shape)
        padded_x_in = np.pad(self.x_in, (self.__n1n2_sinusoids[0], self.__n1n2_sinusoids[1]),
                             'constant', constant_values=(0, 0))
        i_idxs = [j for j in range(self.__n1n2_sinusoids[0], padded_x_in.shape[0]-self.__n1n2_sinusoids[1])]
        n_idxs = [j for j in range(-self.__n1n2_sinusoids[0], self.__n1n2_sinusoids[1])]
        for j, i_idx in enumerate(i_idxs):
            for k, n_idx in enumerate(n_idxs):
                y_out[j] += padded_x_in[(i_idx+n_idx)] * self.gmeds_tap_weights[k][j]

        return y_out

    @staticmethod
    def rescale_factor_for_avg_gain(avg_gain_db, path_complex_coeffs):
        z_pow = np.mean(abs(path_complex_coeffs ** 2))
        x_avg_gain_lin = 10 ** (avg_gain_db / 10)
        omega = np.sqrt(x_avg_gain_lin) / np.sqrt(z_pow)
        return omega
