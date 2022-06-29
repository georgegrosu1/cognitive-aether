import pywt
import numpy as np
from pathlib import Path
from scipy import special as sp
from scipy.signal import convolve2d
from skimage.color import rgba2rgb, rgb2gray
from skimage.measure import shannon_entropy
from skimage.filters import threshold_otsu
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


def shannon_entropy1d(x):
    p = np.histogram(x, density=True, bins='fd')[0]
    return -np.sum(p*np.log2(p))


def shannon_entropy2d(x):
    if x.shape[-1] == 4:
        x = rgb2gray(rgba2rgb(x))
    elif x.shape[-1] == 3:
        x = rgb2gray(x)
    return shannon_entropy(x)


def fractal_dimension(array, max_box_size=None, min_box_size=3, n_samples=30, n_offsets=5):
    """Calculates the fractal dimension of a numpy array.
    Args:
        array (np.ndarray): The array to calculate the fractal dimension of.
        max_box_size (int): The largest box size, given as the power of 2 so that
                            2**max_box_size gives the sidelength of the largest box.
        min_box_size (int): The smallest box size, given as the power of 2 so that
                            2**min_box_size gives the sidelength of the smallest box.
                            Default value 1.
        n_samples (int): number of scales to measure over.
        n_offsets (int): number of offsets to search over to find the smallest set N(s) to
                       cover  all voxels>0.
    """

    # Make image to binary
    if array.shape[-1] == 4:
        array = rgb2gray(rgba2rgb(array))
    elif array.shape[-1] == 3:
        array = rgb2gray(array)
    mask_threshold = threshold_otsu(array)
    array = array >= mask_threshold

    if max_box_size is None:
        # default max size is the largest power of 2 that fits in the smallest dimension of the array:
        max_box_size = int(np.floor(np.log2(np.min(array.shape))))
    scales = np.floor(np.logspace(max_box_size, min_box_size, num=n_samples, base=2))
    scales = np.unique(scales)  # remove duplicates that could occur as a result of the floor

    # get the locations of all non-zero pixels
    locs = np.where(array > 0)
    if len(locs) == 3:
        voxels = np.array([(x, y, z) for x, y, z in zip(*locs)])
    elif len(locs) == 2:
        voxels = np.array([(x, y) for x, y in zip(*locs)])
    elif len(locs) == 1:
        voxels = np.array([x for x in zip(*locs)])
    else:
        raise NotImplementedError

    # count the minimum amount of boxes touched
    ns = []
    # loop over all scales
    for scale_val in scales:
        touched = []
        if n_offsets == 0:
            offsets = np.array([0])
        else:
            offsets = np.linspace(0, scale_val, n_offsets)
        # search over all offsets
        for offset in offsets:
            bin_edges = [np.arange(0, i, scale_val) for i in array.shape]
            bin_edges = [np.hstack([0 - offset, x + offset]) for x in bin_edges]
            h_1, e = np.histogramdd(voxels, bins=bin_edges)
            touched.append(np.sum(h_1 > 0))
        ns.append(touched)
    ns = np.array(ns)

    # From all sets N found, keep the smallest one at each scale
    ns = np.min(ns, axis=1)

    # Only keep scales at which Ns changed
    scales = np.array([np.min(scales[ns == x]) for x in np.unique(ns)])

    ns = np.unique(ns)
    ns = ns[ns > 0]
    scales = scales[:len(ns)]
    # perform fit
    coeffs = np.polyfit(np.log(1 / scales), np.log(ns), 1)

    return coeffs[0]


def lacunarity(image, box_size):
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
    if image.shape[-1] == 4:
        image = rgb2gray(rgba2rgb(image))
    elif image.shape[-1] == 3:
        image = rgb2gray(image)
    mask_threshold = threshold_otsu(image)
    image = image >= mask_threshold
    accumulator = convolve2d(image, kernel, mode='same')
    mean_sqrd = np.mean(accumulator) ** 2
    if mean_sqrd == 0:
        return 0.0

    return np.var(accumulator) / mean_sqrd + 1


def image2double(image):
    im_info = np.finfo(image.dtype)  # Get the data type of the input image
    return image.astype(np.float) / im_info.max  # Divide all values by the largest possible value in the datatype


def spatial_frequency(image):
    # Calculation of spatial frequency according to Li et al., 2001; Eskicioglu and Fisher, 1995.
    # Vectorized implementation

    # Make image to gray and double
    if image.shape[-1] == 4:
        image = rgb2gray(rgba2rgb(image))
    elif image.shape[-1] == 3:
        image = rgb2gray(image)
    image = image2double(image)

    # Compute sum row freq
    sum_row_f = np.sum([image[:, 1:] - image[:, :-1] ** 2])

    # Compute sum col freq
    sum_col_f = np.sum([image[1:, :] - image[:-1, :] ** 2])

    # Get mean square root
    row_f = np.sqrt(sum_row_f / (image.shape[0] * image.shape[1]))
    col_f = np.sqrt(sum_col_f / (image.shape[0] * image.shape[1]))

    return np.sqrt(row_f ** 2 + col_f ** 2)


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


def get_snr_context_rescale_factor(x_in, noise, rx_snr, verbose=False):
    sigma = 10 ** (rx_snr / 10)
    noise_power = np.mean(abs(noise ** 2))
    req_sgn_power = sigma * noise_power
    initial_sgn_power = np.mean(abs(x_in) ** 2)
    if initial_sgn_power == 0:
        return 1, noise_power, req_sgn_power
    factor = np.sqrt(req_sgn_power) / np.sqrt(initial_sgn_power)
    if verbose:
        print(f'Required signal power: {req_sgn_power} [W]=[V^2]'
              f'\nInitial signal power: {initial_sgn_power} [W]=[V^2]'
              f'\nSignal amplitude rescale factor: {factor} [Volts]')
    return factor, noise_power, req_sgn_power
