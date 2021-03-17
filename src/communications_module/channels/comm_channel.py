import numpy as np
from .MetaChannel import MetaChannel


class ChannelModel(MetaChannel):

    def __init__(self, total_carriers_over_ch, channel_type='slow_fading', scaled_ch=True, velocity=None, fc=None,
                 sim_sample_rante=None, number_paths=None, k_rice=None, r_hat_rice=None):
        super().__init__(channel_type)
        self.total_carriers_ch = total_carriers_over_ch
        self.channel_type = channel_type
        self.scaled_ch = scaled_ch
        self.velocity = velocity
        self.fc = fc
        self.sim_sample_rate = sim_sample_rante
        self.number_paths = number_paths
        self.k_rice = k_rice
        self.r_hat_rice = r_hat_rice
        self.ch_response = self._init_channel()

    def _init_channel(self):

        def scale(x, out_range=(0, 1), axis=None):
            domain = np.min(x, axis), np.max(x, axis)
            y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
            return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2

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
        arr_len = int(np.random.uniform(3, 10))
        for coef in range(arr_len + 1):
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
        # :param velocity: Velocity of RX or TX in Km/h
        # :param central_freq: Central frequency
        # :param sim_sample_rate: Sample rate of simulation
        # :param number_paths: Number of paths to sum
        """
        assert (self.velocity is not None) & (self.fc is not None) & (self.sim_sample_rate is not None) \
               & (self.number_paths is not None), 'Make sure class attributes are not None'
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

        return z

    def rician_multipath_fading_channel(self):
        # For self.k_rice=0 this model is theoretically equivalent to Rayleigh fading model

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
            gaussians = np.random.default_rng().normal(mean, sigma, fs)
            return gaussians

        def complex_multipath_fading(r_hat, k_rice, fs):
            # complex_Multipath_Fading generates the complex fading random variables
            p, q = calculate_means(r_hat, k_rice)
            sigma = scattered_component(r_hat, k_rice)
            multipath_fading = generate_gaussian_noise(p, sigma, fs) + (1j * generate_gaussian_noise(q, sigma, fs))
            return multipath_fading

        return complex_multipath_fading(self.r_hat_rice, self.k_rice, self.sim_sample_rate)

    @staticmethod
    def get_linear_ch_magnitude(ch_resp):
        return np.abs(ch_resp)

    @staticmethod
    def get_db_ch_magnitude(ch_resp):
        z_mag = np.abs(ch_resp)
        return 10 * np.log10(z_mag)
