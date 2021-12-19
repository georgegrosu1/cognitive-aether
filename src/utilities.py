import pywt
import numpy as np
from pathlib import Path
from scipy import special as sp
from scipy.signal import convolve2d
from skimage.color import rgba2rgb, rgb2gray
from skimage.restoration import estimate_sigma, denoise_wavelet


def get_abs_path(relative_path) -> Path:
    root_path = Path(__file__).resolve().parent.parent
    final_path = Path(str(root_path) + f'/{relative_path}')
    return final_path


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


def lacunarity(image, box_size, mask_threshold=0.1):
    """
    Calculate the lacunarity value over an image.

    The calculation is performed following these papers:

    Kit, Oleksandr, and Matthias Luedeke. "Automated detection of slum area
    change in Hyderabad, India using multitemporal satellite imagery."
    ISPRS journal of photogrammetry and remote sensing 83 (2013): 130-137.

    Kit, Oleksandr, Matthias Luedeke, and Diana Reckien. "Texture-based
    identification of urban slums in Hyderabad, India using remote sensing
    data." Applied Geography 32.2 (2012): 660-667.
    """
    kernel = np.ones((box_size, box_size))
    image_gray = rgb2gray(rgba2rgb(image))
    binary_img = image_gray >= mask_threshold
    accumulator = convolve2d(binary_img, kernel, mode='valid')
    mean_sqrd = np.mean(accumulator) ** 2
    if mean_sqrd == 0:
        return 0.0

    return np.var(accumulator) / mean_sqrd + 1


def scale(x, out_range=(0, 1), domain: tuple = None, axis=None):
    if domain is None:
        domain = np.min(x, axis), np.max(x, axis)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2


def draw_from_distribution(mean: float = 0, sigma: float = 1, samples: int = 1):
    assert sigma != 0, "Sigma can not be 0"
    z = np.random.standard_normal((samples*2,)).view(np.complex)
    z = sigma * ((z.real + mean) + (z.imag * 1j))
    return z


def get_bayes_denoised(x, noise):
    sigma_est = estimate_sigma(noise, average_sigmas=True)
    rx_bayes = denoise_wavelet(x, method='BayesShrink', mode='soft',
                               sigma=2*sigma_est / (2 - sigma_est), rescale_sigma=True)

    return rx_bayes


def get_visu_denoised(x, noise):
    sigma_est = estimate_sigma(noise, average_sigmas=True)
    rx_visu = denoise_wavelet(x, method='VisuShrink', mode='soft',
                              sigma=2*sigma_est / (1 + sigma_est * 20), rescale_sigma=True)

    return rx_visu


def get_db_magnitude(linear_response):
    z_mag = np.abs(linear_response)
    return 10 * np.log10(z_mag**2)


def snr_db_to_linear(snr_db):
    return 10 ** (snr_db / 10)


def get_snr_context_rescale_factor(x_in, noise, rx_snr):
    sigma = 10 ** (rx_snr / 10)
    noise_power = np.mean(abs(noise ** 2))
    req_sgn_power = sigma * noise_power
    initial_sgn_power = np.mean(abs(x_in ** 2))
    if initial_sgn_power == 0:
        return 1, noise_power, req_sgn_power
    factor = np.sqrt(req_sgn_power) / np.sqrt(initial_sgn_power)
    print(f'Required signal power: {req_sgn_power} [W]=[V^2]'
          f'\nInitial signal power: {initial_sgn_power} [W]=[V^2]'
          f'\nSignal amplitude rescale factor: {factor} [Volts]')
    return factor, noise_power, req_sgn_power
