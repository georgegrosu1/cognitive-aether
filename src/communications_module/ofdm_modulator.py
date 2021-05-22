import numpy as np
from sympy.combinatorics.graycode import GrayCode


class OFDMModulator:

    def __init__(self,
                 bits_per_sym: int,
                 subcarriers: int,
                 cp_ratio_numitor: float,
                 num_pilots: int):

        self.subcarriers = subcarriers
        self.cp_ratio_numitor = cp_ratio_numitor
        self.num_pilots = num_pilots
        self.bits_per_sym = bits_per_sym
        self.cyclic_prefix = self.subcarriers // self.cp_ratio_numitor
        self.ofdm_sym_len = self.subcarriers + self.cyclic_prefix
        self.pilot_default = 3 + 3j
        self.subcarriers_idxs = self.get_subcarriers_idxs()
        self.pilots_idxs = self.get_pilots_idxs()
        self.data_carriers_idxs = np.delete(self.subcarriers_idxs, self.pilots_idxs)
        self.payload_per_ofdm = len(self.data_carriers_idxs) * self.bits_per_sym
        # number of payload bits per OFDM symbol
        self.mapping_table = self.init_mapping_table()

    def get_subcarriers_idxs(self):
        return np.arange(self.subcarriers)

    def get_pilots_idxs(self):
        pilots_idxs = self.subcarriers_idxs[::self.subcarriers // self.num_pilots]
        pilots_idxs = np.hstack([pilots_idxs, np.array([self.subcarriers_idxs[-1]])])
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

    # TODO: Remove when sure its not required anymore
    def init_channel_response(self):
        ch_response = np.array([0.1, 0.2+0.17j, 0.1+0.3j, 0.412+0.051j])
        return np.fft.fft(ch_response, self.subcarriers)

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
        symbol = np.zeros(self.subcarriers, dtype=complex)  # the overall K subcarriers
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

    def generate_ofdm_tx_signal(self, ofdm_symbols: int, continuous_transmission: bool = False):
        ofdm_tx_signal = np.array([])
        for _ in range(ofdm_symbols):
            ofdm_sym = self.generate_ofdm_symbol(continuous_transmission)
            ofdm_tx_signal = np.append(ofdm_sym, ofdm_tx_signal, axis=0)

        return ofdm_tx_signal

    def remove_cyclic_prefix(self, rx_ofdm_signal):
        return rx_ofdm_signal[self.cyclic_prefix:(self.cyclic_prefix + self.subcarriers)]
