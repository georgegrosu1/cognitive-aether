import numpy as np
from sympy.combinatorics.graycode import GrayCode


class OFDMModulator:

    def __init__(self,
                 fft_size: int,
                 bits_per_sym: int,
                 subcarriers: int,
                 cp_ratio_numitor: float,
                 num_pilots: int):

        assert fft_size >= subcarriers, 'FFT points must be greater than active sub-carriers'

        self.fft_size = fft_size
        self.subcarriers = subcarriers
        self.cp_ratio_numitor = cp_ratio_numitor
        self.num_pilots = num_pilots
        self.bits_per_sym = bits_per_sym
        self.cyclic_prefix = self.fft_size // self.cp_ratio_numitor
        self.ofdm_sym_len = self.fft_size + self.cyclic_prefix
        self.pilot_default = 3 + 3j
        self.subcarriers_idxs = self.get_active_subcarriers_idxs()
        self.pilots_idxs = self.get_pilots_idxs()
        self.data_carriers_idxs = self.subcarriers_idxs[~np.in1d(self.subcarriers_idxs, self.pilots_idxs)]
        self.payload_per_ofdm = len(self.data_carriers_idxs) * self.bits_per_sym
        # number of payload bits per OFDM symbol
        self.mapping_table = self.init_mapping_table()

    def _get_active_subcarriers_nums(self):
        if self.subcarriers == self.fft_size:
            return self.get_subcarriers_idxs()

        half_used_subc = self.subcarriers // 2
        first_half = np.r_[1:(half_used_subc + 1)]
        second_half = np.r_[-half_used_subc:0]

        return np.hstack([first_half, second_half])

    def get_active_subcarriers_idxs(self):
        numbers = self._get_active_subcarriers_nums()
        half_used = self.subcarriers // 2

        indexes_proper = np.hstack(
            [self.fft_size + numbers[half_used:], numbers[0:half_used]])
        return indexes_proper

    def get_subcarriers_idxs(self):
        return np.fft.fftshift(np.arange(self.fft_size) - self.fft_size // 2)

    def get_pilots_idxs(self):
        pilots_idxs = self.subcarriers_idxs[::int(np.ceil(self.subcarriers / self.num_pilots))]
        return pilots_idxs

    def check_perfect_square(self):
        return np.sqrt(np.power(2, self.bits_per_sym)).is_integer()

    def compute_amps_list(self):
        assert self.check_perfect_square(), 'Num of bits must be a square of 2'
        neg_amps, pos_amps = [], []
        amp = 1
        step = 2
        for _ in range(int(np.sqrt(np.power(2, self.bits_per_sym)) / 2)):
            neg_amps.append(-amp)
            pos_amps.append(amp)
            amp += step
        return sorted(neg_amps + pos_amps)

    def serial_2_parallel(self, bits_array):
        return bits_array.reshape((len(self.data_carriers_idxs), self.bits_per_sym))

    def generate_payload(self):
        ignore_flag = np.random.randint(low=0, high=2, size=1)[0]
        bits = np.random.binomial(n=1, p=0.5, size=(self.payload_per_ofdm,))
        payload_words = self.serial_2_parallel(bits)
        return payload_words, ignore_flag

    def map_words_2_qam(self, payload):
        return np.array([self.mapping_table[str(b).replace(' ', '').replace('[', '').replace(']', '')]
                         for b in payload])

    def init_mapping_table(self):
        assert self.check_perfect_square(), 'Num of bits must be a square of 2'
        mapping_table = {}
        graycode_words = list(GrayCode(self.bits_per_sym).generate_gray())
        re_amps_list, im_amps_list = self.compute_amps_list(), self.compute_amps_list()
        im_amp_idx = 0
        re_amps = re_amps_list
        for idx, word in enumerate(graycode_words):
            if idx % len(re_amps_list) == 0 and idx != 0:
                im_amp_idx += 1
                re_amps = re_amps[::-1]
            mapping_table.setdefault(word, re_amps[idx % len(re_amps_list)] + im_amps_list[im_amp_idx]*1j)

        return mapping_table

    def payload_and_pilots_mapping(self, qam_payload, ioja):
        symbol = np.zeros(self.fft_size, dtype=complex)  # the overall K subcarriers
        if not bool(ioja):
            symbol[self.pilots_idxs] = self.pilot_default  # allocate the pilot subcarriers
            symbol[self.data_carriers_idxs] = qam_payload  # allocate the data subcarriers
        return symbol

    @staticmethod
    def ofdm_idft(ofdm_symbol_data):
        return np.fft.ifft(ofdm_symbol_data)

    def add_cyclic_prefix(self, ofdm_symbol_time_domain):
        cp = ofdm_symbol_time_domain[-self.cyclic_prefix:]  # take the last CP samples ...
        return np.hstack([cp, ofdm_symbol_time_domain])  # ... and add them to the beginning

    def generate_ofdm_symbol(self, force_flag: bool = True):
        payload, flag = self.generate_payload()
        if force_flag:
            flag = False
        qam_load = self.map_words_2_qam(payload)
        ofdm_mapping = self.payload_and_pilots_mapping(qam_load, flag)
        ofdm_ift_time_domain = self.ofdm_idft(ofdm_mapping)
        ofdm_w_cp = self.add_cyclic_prefix(ofdm_ift_time_domain)
        return ofdm_w_cp

    def generate_ofdm_tx_signal(self, ofdm_symbols: int,
                                continuous_transmission: bool = False,
                                continuous_silence: bool = False):
        assert (continuous_transmission and continuous_silence) is not True, \
            'Continuous transmission and silence can not be True at the same time'
        ofdm_tx_signal = np.array([])
        for _ in range(ofdm_symbols):
            ofdm_sym = self.generate_ofdm_symbol(continuous_transmission)
            if continuous_silence is True:
                ofdm_sym -= ofdm_sym
            ofdm_tx_signal = np.append(ofdm_sym, ofdm_tx_signal, axis=0)

        return ofdm_tx_signal

    def remove_cyclic_prefix(self, rx_ofdm_signal):
        return rx_ofdm_signal[self.cyclic_prefix:(self.cyclic_prefix + self.subcarriers)]
