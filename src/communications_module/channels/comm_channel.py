import numpy as np
from .MetaChannel import MetaChannel

from src.utilities import scale


class ChannelModel(MetaChannel):

    def __init__(self, total_carriers_over_ch, channel_type='slow_fading', scaled_ch=True, velocity=None, fc=None,
                 number_paths=None, k_rice=None, r_hat_rice=None):
        super().__init__(channel_type)
        self.total_carriers_ch = total_carriers_over_ch
        self.channel_type = channel_type
        self.scaled_ch = scaled_ch
        self.velocity = velocity
        self.fc = fc
        self.sim_sample_rate = total_carriers_over_ch
        self.number_paths = number_paths
        self.k_rice = k_rice
        self.r_hat_rice = r_hat_rice
        self.ch_response = self._init_channel()

    def _init_channel(self):

        def get_channel_by_type():
            ch_fading_dict = {
                'slow_fading': self.slow_fading_channel(),
                'rayleigh_fading': self.rayleigh_multipath_fading_channel(),
                'rician_fading': self.rician_multipath_fading_channel()
            }
            if self.scaled_ch:
                return scale(ch_fading_dict[self.channel_type])
            return ch_fading_dict[self.channel_type]

        return get_channel_by_type()

    def slow_fading_channel(self):
        arr = []
        for coef in range(self.total_carriers_ch + 1):
            ignore_flag = np.random.randint(low=0, high=2, size=1)[0]
            re, im = np.random.uniform(low=0.0, high=1.0), np.random.uniform(low=0.0, high=1.0)
            w = re + complex(im)
            if ignore_flag:
                arr.append(re)
            else:
                arr.append(w)
        channel_response = np.array(arr)  # the impulse response of the wireless channel
        return np.fft.fft(channel_response, self.total_carriers_ch)

    def rayleigh_multipath_fading_channel(self):
        """
        # :param velocity: Velocity of RX/tx relative to TX/rx in km/h
        # :param central_freq: Central frequency
        # :param sim_sample_rate: Sample rate of simulation
        # :param number_paths: Number of paths to sum
        """
        assert (self.velocity is not None) & (self.fc is not None) & (self.sim_sample_rate is not None) \
               & (self.number_paths is not None), 'Make sure class attributes are not None'
        z_init = [1, 0]
        fc = self.fc  # RF carrier frequency in Hz
        fs = self.sim_sample_rate  # sample rate of simulation
        num_paths = self.number_paths  # number of sinusoids to sum
        v_ms = self.velocity * 1000 / 3600  # convert to m/s
        fd = v_ms * fc / 3e8  # max Doppler shift
        t = np.arange(0, 1, 1 / fs)  # time vector. (start, stop, step)
        x = np.zeros(len(t))
        y = np.zeros(len(t))
        for i in range(num_paths):
            alpha = (np.random.rand() - 0.5) * 2 * np.pi
            phi = (np.random.rand() - 0.5) * 2 * np.pi
            x += np.random.randn() * np.cos(2 * np.pi * fd * t * np.cos(alpha) + phi)
            y += np.random.randn() * np.sin(2 * np.pi * fd * t * np.cos(alpha) + phi)

        # z is the complex coefficient representing channel, you can think of this as a phase shift and magnitude scale
        z = (1 / np.sqrt(num_paths)) * (x + 1j * y)  # this is what you would actually use when simulating the channel
        z = np.hstack([z_init, z])
        return np.fft.fft(z, self.total_carriers_ch)

    def rician_multipath_fading_channel(self):
        # For self.k_rice=0 this model is theoretically equivalent to Rayleigh fading model
        assert (self.k_rice is not None) & (self.r_hat_rice is not None), 'Make sure class attributes are not None'

        def calculate_means(r_hat, k_rice):
            # calculate_means calculates the means of the complex Gaussians representing the
            # in-phase and quadrature components
            phi = (np.random.rand() - 0.5) * 2 * np.pi
            p = np.sqrt(k_rice * r_hat / (1 + r_hat)) * np.cos(phi)
            q = np.sqrt(k_rice * r_hat / (1 + r_hat)) * np.sin(phi)
            return p, q

        def scattered_component(r_hat, k_rice):
            sigma = np.sqrt(r_hat / (2 * (1 + k_rice)))
            return sigma

        def generate_gaussian_noise(mean, sigma, fs):
            # generate_Gaussians generates the Gaussian random variables
            gaussians = np.random.default_rng().normal(mean, sigma, int(fs))
            return gaussians

        def complex_multipath_fading(r_hat, k_rice, fs):
            # complex_Multipath_Fading generates the complex fading random variables
            z_init = [1, 0]
            p, q = calculate_means(r_hat, k_rice)
            sigma = scattered_component(r_hat, k_rice)
            multipath_fading = generate_gaussian_noise(p, sigma, fs) + (1j * generate_gaussian_noise(q, sigma, fs))
            z = np.hstack([z_init, multipath_fading])
            return np.fft.fft(z, self.total_carriers_ch)

        return complex_multipath_fading(self.r_hat_rice, self.k_rice, self.sim_sample_rate)

    def tapped_delay_channel(self, delays: list, fading_type: str):
        raise NotImplemented

    @staticmethod
    def draw_from_distribution(num_draws: int, mean: int = 0, sigma: float = 1, samples: int = 10000):
        assert sigma != 0, "Sigma can not be 0"
        samples = num_draws * (num_draws > samples) + samples * (num_draws <= samples)
        i = np.random.standard_normal((samples,)) * sigma + mean
        q = np.random.standard_normal((samples,)) * sigma
        z = [complex(i[idx], q[idx]) for idx in range(len(i))]
        drawn = [z[np.random.randint(0, len(z) - 1)] for _ in range(num_draws)]
        return np.abs(drawn)

    @staticmethod
    def get_linear_ch_magnitude(ch_resp):
        return np.abs(ch_resp)

    @staticmethod
    def get_db_ch_magnitude(ch_resp):
        z_mag = np.abs(ch_resp)
        return 10 * np.log10(z_mag)
