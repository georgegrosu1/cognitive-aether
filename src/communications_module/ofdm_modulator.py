import numpy as np
from src.communications_module.ofdm_base import OFDMBase


class OFDMModulator(OFDMBase):

    def __init__(self, fft_size: int, bits_per_sym: int, subcarriers: int, cp_ratio_numitor: float, num_pilots: int):
        super().__init__(fft_size, bits_per_sym, subcarriers, cp_ratio_numitor, num_pilots)

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

    def payload_and_pilots_mapping(self, qam_payload, user_data_carriers, user_pilots_carriers, ioja,
                                   ofdm_sym_freq_domain=None):
        if ofdm_sym_freq_domain is None:
            ofdm_sym_freq_domain = np.zeros(self.fft_size, dtype=complex)  # the overall K subcarriers
        if not bool(ioja):
            ofdm_sym_freq_domain[user_pilots_carriers.astype(int)] = self.pilot_default  # allocate the pilot subcarriers
            ofdm_sym_freq_domain[user_data_carriers.astype(int)] = qam_payload  # allocate the data subcarriers
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
            for user_data_carriers, user_pilots in zip(self.data_carriers_idxs, self.pilots_idxs.squeeze()):
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
