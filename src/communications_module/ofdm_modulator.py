import numpy as np
from typing import Union
from sympy.combinatorics.graycode import GrayCode


class OFDMModulator:

    def __init__(self,
                 fft_size: int,
                 bits_per_sym: int,
                 subcarriers: Union[int, list, np.ndarray],
                 cp_ratio_numitor: float,
                 num_pilots: Union[int, list]):

        self.fft_size = fft_size
        if (type(subcarriers) is np.ndarray) | (type(subcarriers) is list):
            if type(subcarriers) is list:
                if any(len(user) != subcarriers[0] for user in subcarriers[1:]):
                    subc_renew = []
                    for user in subcarriers:
                        subc_renew.append(np.array(user))
                    subcarriers = subc_renew
            self.subcarriers = np.array(subcarriers, dtype=object)
            assert self.subcarriers.ndim < 3, 'Subcarriers must have maximum 2 dimension'
            assert fft_size >= np.sum(user.shape[0] for user in self.subcarriers), \
                'FFT points must be greater than active sub-carriers'
            assert type(num_pilots) is list, 'Provide number of pilots in list form'
            assert len(num_pilots) == self.subcarriers.shape[0], 'Provide number of pilots for each user'
            for user, user_pilots in zip(self.subcarriers, num_pilots):
                assert user_pilots <= user.shape[0], 'Number of user pilots can not be greater than # of subcarriers'
            if self.subcarriers.shape[0] > 1:
                for user in range(self.subcarriers.shape[0]):
                    assert type(self.subcarriers[user]) == np.ndarray, 'Subcarriers must contain ndarrays'
                    assert self.subcarriers[user].shape[0] == len(set(self.subcarriers[user])), \
                        'Subcarriers indexes array must not have duplicates'
                    for next_user in range(user+1, self.subcarriers.shape[0]):
                        assert (not(any(np.intersect1d(self.subcarriers[user], self.subcarriers[next_user])))), \
                            'Multiple users must not share same subcarrier indexes'
        else:
            self.subcarriers = subcarriers
            assert (type(num_pilots) is int) & (num_pilots <= self.subcarriers), \
                'Number of user pilots must be an integer and can not be greater than # of subcarriers'
            assert fft_size >= self.subcarriers, 'FFT points must be greater than active sub-carriers'
        self.cp_ratio_numitor = cp_ratio_numitor
        self.num_pilots = num_pilots
        self.bits_per_sym = bits_per_sym
        self.cyclic_prefix = self.fft_size // self.cp_ratio_numitor
        self.ofdm_sym_len = self.fft_size + self.cyclic_prefix
        self.pilot_default = 3 + 3j
        self.subcarriers_idxs = self.get_active_subcarriers_idxs()
        self.pilots_idxs = self.get_pilots_idxs()
        if type(num_pilots) is list:
            self.data_carriers_idxs = []
            for user_carriers, user_pilots in zip(self.subcarriers_idxs, self.pilots_idxs):
                self.data_carriers_idxs.append(user_carriers[~np.in1d(user_carriers, user_pilots)])
            self.data_carriers_idxs = np.array(self.data_carriers_idxs, dtype=object)
        else:
            self.data_carriers_idxs = np.array([self.subcarriers_idxs[~np.in1d(self.subcarriers_idxs,
                                                                               self.pilots_idxs)]])
        self.payload_per_ofdm = np.array([user_data_carriers_idxs.shape[0] * self.bits_per_sym
                                          for user_data_carriers_idxs in self.data_carriers_idxs])
        # number of payload bits per OFDM symbol
        self.mapping_table = self.init_mapping_table()

    def _get_active_subcarriers_nums(self):
        assert (type(self.subcarriers) is int) | (type(self.subcarriers) is np.ndarray), \
            "Provide number of active subcarriers or a numpy array of selection between [-NFFT/2, 0), (0, NFFT/2]"
        if type(self.subcarriers) == int:
            if self.subcarriers == self.fft_size:
                return self.get_subcarriers_idxs()

            half_used_subc = self.subcarriers // 2
            first_half = np.r_[1:(half_used_subc + 1)]
            second_half = np.r_[(-half_used_subc) - (self.subcarriers % 2):0]

            return np.hstack([first_half, second_half])

        return self.subcarriers

    def get_active_subcarriers_idxs(self):
        numbers = self._get_active_subcarriers_nums()
        if type(self.subcarriers) is np.ndarray:
            return numbers

        half_used = self.subcarriers // 2
        indexes_proper = np.hstack(
            [self.fft_size + numbers[half_used:], numbers[0:half_used]])

        return indexes_proper

    def get_subcarriers_idxs(self):
        idxs = np.fft.fftshift(np.arange(self.fft_size) - (self.fft_size // 2) - (self.fft_size % 2) + 1)[:-1]
        idxs = np.insert(idxs, np.where(idxs == np.min(idxs))[0], min(idxs)-1)
        return idxs

    def get_pilots_idxs(self):
        if type(self.subcarriers) is int:
            return np.array(self.subcarriers_idxs[::int(np.ceil(self.subcarriers / self.num_pilots))])
        pilot_idxs = []
        for user, user_pilots in zip(self.subcarriers_idxs, self.num_pilots):
            pilot_idxs.append(user[::int(np.ceil(user.shape[0] / user_pilots))])

        return np.array(pilot_idxs, dtype=object)

    def check_perfect_square(self):
        return np.sqrt(np.power(2, self.bits_per_sym)).is_integer()

    def compute_amps_list(self):
        neg_amps, pos_amps = [], []
        amp = 1
        step = 2
        for _ in range(int(np.sqrt(np.power(2, self.bits_per_sym)) / 2)):
            neg_amps.append(-amp)
            pos_amps.append(amp)
            amp += step
        return sorted(neg_amps + pos_amps)

    def serial_2_parallel(self, bits_array, user_data_carriers):
        return bits_array.reshape((len(user_data_carriers), self.bits_per_sym))

    def generate_payload(self, user_data_carriers):
        ignore_flag = np.random.randint(low=0, high=2, size=1)[0]
        user_pay_idx = [np.sum(ref_carriers == user_data_carriers) == ref_carriers.shape[0]
                        for ref_carriers in self.data_carriers_idxs]
        bits = np.random.binomial(n=1, p=0.5, size=(self.payload_per_ofdm[user_pay_idx][0],))
        payload_words = self.serial_2_parallel(bits, user_data_carriers)
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

    def payload_and_pilots_mapping(self, qam_payload, user_data_carriers, user_pilots_carriers, ioja,
                                   ofdm_sym_freq_domain=None):
        if ofdm_sym_freq_domain is None:
            ofdm_sym_freq_domain = np.zeros(self.fft_size, dtype=complex)  # the overall K subcarriers
        if not bool(ioja):
            ofdm_sym_freq_domain[user_pilots_carriers] = self.pilot_default  # allocate the pilot subcarriers
            ofdm_sym_freq_domain[user_data_carriers] = qam_payload  # allocate the data subcarriers
        return ofdm_sym_freq_domain

    @staticmethod
    def ofdm_idft(ofdm_symbol_data):
        return np.fft.ifft(ofdm_symbol_data)

    def add_cyclic_prefix(self, ofdm_symbol_time_domain):
        cp = ofdm_symbol_time_domain[-self.cyclic_prefix:]  # take the last CP samples ...
        return np.hstack([cp, ofdm_symbol_time_domain])  # ... and add them to the beginning

    def generate_ofdm_symbol(self, user_data_carriers, user_pilots_carriers, ofdm_sym_freq_domain=None,
                             force_flag: bool = True):
        payload, flag = self.generate_payload(user_data_carriers)
        if force_flag:
            flag = False
        qam_load = self.map_words_2_qam(payload)
        ofdm_sym_freq_domain = self.payload_and_pilots_mapping(qam_load, user_data_carriers, user_pilots_carriers,
                                                               flag, ofdm_sym_freq_domain)
        return ofdm_sym_freq_domain

    def generate_ofdm_tx_signal(self, ofdm_symbols: int,
                                continuous_transmission: bool = False,
                                continuous_silence: bool = False):
        assert (continuous_transmission and continuous_silence) is not True, \
            'Continuous transmission and silence can not be True at the same time'
        ofdm_tx_signal = np.array([])
        for _ in range(ofdm_symbols):
            ofdm_sym_freq_domain = np.zeros(self.fft_size, dtype=complex)
            for user_data_carriers, user_pilots in zip(self.data_carriers_idxs, self.pilots_idxs):
                ofdm_sym_freq_domain = self.generate_ofdm_symbol(user_data_carriers, user_pilots,
                                                                 ofdm_sym_freq_domain, continuous_transmission)
            ofdm_ift_time_domain = self.ofdm_idft(ofdm_sym_freq_domain)
            ofdm_sym = self.add_cyclic_prefix(ofdm_ift_time_domain)
            if continuous_silence is True:
                ofdm_sym -= ofdm_sym
            ofdm_tx_signal = np.append(ofdm_sym, ofdm_tx_signal, axis=0)

        return ofdm_tx_signal

    def remove_cyclic_prefix(self, rx_ofdm_signal):
        return rx_ofdm_signal[self.cyclic_prefix:(self.cyclic_prefix + self.subcarriers)]
