import numpy as np


ALL_RX_FEATURES = {f'{snr}_RX_OFDM' for snr in np.arange(-50, 50, 0.5)}
ALL_BAYESSHRINK_FEATURES = {f'{snr}_RX_OFDM_BayesShrink' for snr in np.arange(-50, 50, 0.5)}
ALL_VISUSHRINK_FEATURES = {f'{snr}_RX_OFDM_VisuShrink' for snr in np.arange(-50, 50, 0.5)}
