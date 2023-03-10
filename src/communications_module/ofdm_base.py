import numpy as np
from typing import Union
from sympy.combinatorics.graycode import GrayCode


class OFDMBase:

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
        self.pilots_idxs = np.array([self.get_pilots_idxs()])
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
            sorted_subcarriers = np.sort(self.subcarriers_idxs)
            return np.array(sorted_subcarriers[::int(np.ceil(self.subcarriers / self.num_pilots))+1])
        pilot_idxs = []
        for user, user_pilots in zip(self.subcarriers_idxs, self.num_pilots):
            pilot_idxs.append(np.sort(user)[::int(np.ceil(user.shape[0] / user_pilots))+1])

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

    def add_cyclic_prefix(self, ofdm_symbol_time_domain):
        cp = ofdm_symbol_time_domain[-self.cyclic_prefix:]  # take the last CP samples ...
        return np.hstack([cp, ofdm_symbol_time_domain])  # ... and add them to the beginning

    def remove_cyclic_prefix(self, rx_ofdm_signal):
        return rx_ofdm_signal[self.cyclic_prefix:(self.cyclic_prefix + self.subcarriers)]
