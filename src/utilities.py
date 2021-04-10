import pywt
import numpy as np
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
