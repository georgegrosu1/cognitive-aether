import pywt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import special as sp
from skimage.restoration import estimate_sigma, denoise_wavelet


def invqfunc(x):
    return np.sqrt(2)*sp.erfinv(1-2*x)


def window_pow(x):
    return np.mean(x**2)


def window_pow_db(x):
    return np.log10(np.mean(x**2))


def pow_c_a_dwt(x):
    c_a, _ = pywt.dwt(x, 'db3')
    return window_pow_db(c_a)


def pow_c_d_dwt(x):
    _, c_d = pywt.dwt(x, 'db3')
    return window_pow_db(c_d)


def logistic_map(x, g_rate=0.6):
    pop = np.sum(np.histogram(x, 'fd', density=True)[1])
    log_map = pop * g_rate * (1 - pop)
    return abs(np.log2(abs(log_map)))


def shannon_entropy(x):
    p = np.histogram(x, density=True, bins='fd')[0]
    return -np.sum(p*np.log2(p))


def scale(x, out_range=(0, 1), axis=None):
    domain = np.min(x, axis), np.max(x, axis)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2


def get_bayes_denoised(x):
    sigma_est = estimate_sigma(x, average_sigmas=True)
    rx_bayes = denoise_wavelet(x, method='BayesShrink', mode='soft',
                               sigma=sigma_est / (1 + sigma_est), rescale_sigma=True)

    return rx_bayes


def get_visu_denoised(x):
    sigma_est = estimate_sigma(x, average_sigmas=True)
    rx_visu = denoise_wavelet(x, method='VisuShrink', mode='soft',
                              sigma=sigma_est / (1 + sigma_est * 20), rescale_sigma=True)

    return rx_visu


def plot_roc(roc_df: pd.DataFrame, compare_signals, sensing_window=2):
    plt.figure(figsize=(15, 10))
    pfs = np.arange(0, 1, 0.01)
    for signal in compare_signals:
        snr_db = int(signal.split('_')[0])
        pds = []
        gt_sgn = f'{snr_db}_TX_OFDM'
        sigma = estimate_sigma(roc_df[signal], average_sigmas=True)
        tps_num = len(np.where(roc_df[gt_sgn] > 0)[0])
        for fals_proba in pfs:
            positiv_cases = 0
            window = sensing_window
            for idx in range(0, len(roc_df[signal])):
                if len(roc_df[signal]) - idx <= window:
                    window -= 1
                energy = abs(roc_df.loc[idx:(idx+window), signal])**2
                fin_energy = 1/sensing_window * sum(energy)
                thresh = sigma*(invqfunc(fals_proba))*(np.sqrt(2*window)+window)
                if (fin_energy >= thresh) and (roc_df.loc[idx, gt_sgn] > 0):
                    positiv_cases += 1
            pd_val = positiv_cases/tps_num
            pds.append(pd_val)
        plt.plot(pfs, pds, label=signal)
    plt.xlim(0, 1.1)
    plt.plot([0, 1], [0, 1], ls="--", c=".3")
    plt.ylabel('Probability of detection (TP)')
    plt.xlabel('Probability of false alarm (FP)')
    plt.title('Energy Detection - Reciever Operating Characteristics')
    plt.legend()
    plt.show()
