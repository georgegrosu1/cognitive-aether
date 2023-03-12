import numpy as np
from scipy.interpolate import interp1d
from src.communications_module.ofdm_base import OFDMBase


class ChannelEq(OFDMBase):
    def __init__(self, fft_size, bits_per_sym, subcarriers, cp_ratio_numitor, num_pilots):
        super().__init__(fft_size, bits_per_sym, subcarriers, cp_ratio_numitor, num_pilots)

    def remove_cyclic_prefix(self, rx_ofdm_symbol):
        return rx_ofdm_symbol[self.cyclic_prefix:]

    @staticmethod
    def ofdm_sym_dft(nocp_ofdm_sym):
        return np.fft.fft(nocp_ofdm_sym)

    def ch_estimation(self, nocp_dft_ofdm_sym):
        pilots_idxs_arr = self.pilots_idxs.squeeze()
        pilots_vals = nocp_dft_ofdm_sym[pilots_idxs_arr]
        h_est = pilots_vals / self.pilot_default

        h_est_abs = interp1d(pilots_idxs_arr, abs(h_est), kind='linear')(self.subcarriers_idxs)
        h_est_phase = interp1d(pilots_idxs_arr, np.angle(h_est), kind='linear')(self.subcarriers_idxs)

        return h_est_abs * np.exp(1j*h_est_phase)

    def eq_ofdm_sym(self, nocp_dft_ofdm_sym):
        ch_est = self.ch_estimation(nocp_dft_ofdm_sym)
        nocp_dft_ofdm_sym[self.subcarriers_idxs] = nocp_dft_ofdm_sym[self.subcarriers_idxs] / ch_est
        return nocp_dft_ofdm_sym

    def get_ofdm_sym_payload(self, ofdm_sym, remove_cp=True, apply_eq=True):
        if remove_cp:
            # Remove cyclic prefix
            ofdm_sym = self.remove_cyclic_prefix(ofdm_sym)

        # Apply DFT
        ofdm_sym_dft = self.ofdm_sym_dft(ofdm_sym)

        if apply_eq:
            # Apply zero focring equalization
            ofdm_sym_dft = self.eq_ofdm_sym(ofdm_sym_dft)

        # Return payload
        return ofdm_sym_dft[self.data_carriers_idxs]

    @staticmethod
    def parallel2serial(parallel_stream):
        return parallel_stream.reshape((-1, ))

    def demapping(self, qam_est):
        # array of possible constellation points
        constellation = np.array([bval for bval in list(self.mapping_table.keys())])
        # constellation = constellation.reshape(constellation.shape[-1]//self.tx_bits_per_sym, self.tx_bits_per_sym)

        # calculate distance of each RX point to each possible point
        dists = abs(qam_est.reshape((-1, 1)) - constellation.reshape((1, -1)))

        # for each element in QAM, choose the index in constellation
        # that belongs to the nearest constellation point
        const_index = dists.argmin(axis=1)

        # get back the real constellation point
        hard_decision = constellation[const_index]
        # str_hardDecision = [str(t)[1:-1].replace(' ', '') for t in hardDecision]

        # transform the constellation point into the bit groups
        return np.vstack([self.mapping_table[i] for i in hard_decision]), hard_decision
