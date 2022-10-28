import json
import uuid
import tqdm
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utilities import window_pow_db, get_bayes_denoised, get_visu_denoised
from src.utilities import pow_c_a_dwt, pow_c_d_dwt, logistic_map
from src.communications_module import OFDMModulator, RandomSampling
from src.communications_module.channels import AWGNChannel, FadingChannel
from src.utilities import get_abs_path
from src.superlets import superlet


def get_saving_dataset_path(configs):
    save_dir = get_abs_path(configs['data_save_path'])
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def get_ofdm_data(ofdm_cfg):
    ofdm_gen = OFDMModulator(bits_per_sym=ofdm_cfg['bits_per_sym'],
                             fft_size=ofdm_cfg['fft_size'],
                             subcarriers=ofdm_cfg['subcarriers'],
                             cp_ratio_numitor=ofdm_cfg['cp_ratio'],
                             num_pilots=ofdm_cfg['num_pilots'])
    ofdm_sign = ofdm_gen.generate_ofdm_tx_signal(ofdm_cfg['num_symbols'],
                                                 continuous_transmission=ofdm_cfg['continuous_transmit'],
                                                 continuous_silence=ofdm_cfg['continuous_silence'])

    return ofdm_sign


def get_channel_faded_data(fading_cfg, x_in_sgn):
    fading_channel = FadingChannel(x_in=x_in_sgn,
                                   channel_type=fading_cfg['type'],
                                   discrete_path_delays=fading_cfg['discrete_del'],
                                   avg_path_gains=fading_cfg['avg_path_gains'],
                                   max_doppler_shift=fading_cfg['max_doppler_shift'],
                                   k_factors=fading_cfg['k_factors'])
    y_out_faded = fading_channel.filter_x_in()

    return y_out_faded


def create_superlet_scalogram(df, superlet_cfg, dir_path, file_name):
    """
    :param df: Pandas:DataFrame based on which the superlet is computed
    :param superlet_cfg: JSON config of superlet
    :param dir_path: Directory path where data is saved
    :param file_name: Name of the dataframe file
    """
    scalograms_dir_false = dir_path / '0'
    scalograms_dir_true = dir_path / '1'
    scalograms_dir_false.mkdir(parents=True, exist_ok=True)
    scalograms_dir_true.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.axis('off')

    # Prepare scalogram params
    target_signal = superlet_cfg['target_signal']
    # NOTE: It is recommended to have the rolling windows smaller or equal to the length of a TX symbol
    roll_window = superlet_cfg['sliding_window']
    roll_step = superlet_cfg['step']
    gt_percent = superlet_cfg['gt_percent']
    foi = np.linspace(superlet_cfg['foi'][0], superlet_cfg['foi'][1], superlet_cfg['foi'][2])
    extent = [0, roll_window / superlet_cfg['samplerate'], foi[0], foi[-1]]
    scales = (1 / foi) / (2 * np.pi)
    label_window = int(gt_percent * roll_window)

    # Generate indexes for sliding window
    start_indexes = [i for i in range(0, df.index[-1] - roll_window, roll_step)]
    end_indexes = [i for i in range(roll_window, df.index[-1], roll_step)]

    for start_idx, end_idx in zip(start_indexes, end_indexes):

        # Get the absolute values fo the Superlet coeffs and generate scalogram
        signal = df.loc[start_idx:end_idx, target_signal]
        spec = superlet(signal,
                        samplerate=superlet_cfg['samplerate'],
                        scales=scales,
                        order_max=superlet_cfg['order_max'],
                        order_min=superlet_cfg['order_min'],
                        c_1=superlet_cfg['c_1'], adaptive=superlet_cfg['adaptive'])
        ampls = np.abs(spec)

        ax.imshow(ampls, cmap="inferno", aspect="auto", extent=extent, origin='lower')

        # Save scalogram in corresponding label directory
        img_name = file_name + f'_IDX_{start_idx}_{uuid.uuid4()}.png'
        gt_lookback_idx = end_idx - label_window
        if np.mean(df['USER'][gt_lookback_idx:end_idx]) >= 1:
            save_path = scalograms_dir_true / img_name
        else:
            save_path = scalograms_dir_false / img_name
        fig.savefig(save_path, dpi=100, transparent=True, bbox_inches='tight', pad_inches=0)
        fig.canvas.flush_events()


def create_features(configs, rx_snr):
    df = pd.DataFrame()
    # Init default noise
    if configs['feature_engineering']['denoising']:
        assert configs['active_channels']['awgn'], "Noise can not be none when applying denoising techniques." \
                                                   "Please activate noise channel"
    disposable_symbols = compute_required_disposable_symbols(configs)
    configs['ofdm_modulator']["num_symbols"] += disposable_symbols

    # Generate TX signal first
    tx_signal = get_ofdm_data(configs['ofdm_modulator'])
    df['RE_TX_OFDM'] = tx_signal.real
    df['IM_TX_OFDM'] = tx_signal.imag
    df['TX_OFDM'] = abs(tx_signal)

    # Init RX signal with TX signal and get rolling window size
    rx_signal = tx_signal
    rolling_window = configs['feature_engineering']['sliding_window_size']

    # Apply fading channel if set
    if configs['active_channels']['fading']:
        rx_signal = get_channel_faded_data(configs['fading_channel'], tx_signal)

    # Apply AWGN channel if set
    if configs['active_channels']['awgn']:
        noise_model = AWGNChannel(rx_signal, rx_snr)
        rx_signal, noise = noise_model.filter_x_in()
        df['RE_NOISE'] = noise.real
        df['IM_NOISE'] = noise.imag
        df['NOISE'] = abs(noise)
        if rolling_window > 0:
            df['sigma'] = df['NOISE'].rolling(rolling_window).apply(np.var)

    df['RX_OFDM'] = abs(rx_signal)
    df['RE_RX_OFDM'] = rx_signal.real
    df['IM_RX_OFDM'] = rx_signal.imag

    # Create denoised signals & extra features
    features_list = configs['feature_engineering']['denoising'] + configs['feature_engineering']['extract_features']
    for feature_technique in features_list:
        upper_technique = feature_technique.upper()
        signal_name = f'RX_{upper_technique}'
        if 'visu' in feature_technique:
            df[signal_name] = get_visu_denoised(df['RX_OFDM'], df['NOISE'])
        elif 'bayes' in feature_technique:
            df[signal_name] = get_bayes_denoised(df['RX_OFDM'], df['NOISE'])
        elif 'pow_db' in feature_technique:
            df[signal_name] = df['RX_OFDM'].rolling(rolling_window).apply(window_pow_db)
        elif 'pow_logistic_map' in feature_technique:
            df[signal_name] = df['RX_BAYES'].rolling(rolling_window).apply(logistic_map)
        elif 'ca_dwt' in feature_technique:
            df[signal_name] = df['RX_OFDM'].rolling(rolling_window).apply(pow_c_a_dwt)
        elif 'cd_dwt' in feature_technique:
            df[signal_name] = df['RX_OFDM'].rolling(rolling_window).apply(pow_c_d_dwt)

    # Generate Ground Truth based on TX signal
    ones = df['TX_OFDM'] > 0
    df.loc[ones, 'USER'] = 1
    df.loc[~ones, 'USER'] = 0

    # Remove the disposable symbols and leave the number specified in configs
    df = df.loc[(configs['ofdm_modulator']['fft_size'] +
                 configs['ofdm_modulator']['fft_size'] // configs['ofdm_modulator']['cp_ratio'] *
                 disposable_symbols):, :]

    configs['ofdm_modulator']["num_symbols"] -= disposable_symbols

    # if use_rand_sampling:
    #     rand_sample = RandomSampling(decimation=decimation_val, sampling_type=samp_type)
    #     rand_samp_idxs = rand_sample.get_sampling_idxs(rx_signal)

    return df


def compute_required_disposable_symbols(configs):
    # Add disposable OFDM symbols based on rolling window size relative to size of OFDM symbol
    win_size = configs['feature_engineering']['sliding_window_size']
    len_cyclic_prefix = configs['ofdm_modulator']['fft_size'] // configs['ofdm_modulator']['cp_ratio']
    ofdm_symbol_size = configs['ofdm_modulator']['fft_size'] + len_cyclic_prefix
    disposable_symbols = int(np.ceil(win_size / ofdm_symbol_size))

    return disposable_symbols


def generate_dataset(dataset_cfg):
    abs_cfg_path = get_abs_path(dataset_cfg)
    with open(abs_cfg_path, 'r') as cfg_file:
        configs = json.load(cfg_file)

    for rx_snr in tqdm.tqdm(configs['awgn_channel']['rx_snrs_list']):
        # Instantiate name of the file
        file_name = f''
        qam_constel = 2 ** int(configs['ofdm_modulator']['bits_per_sym'])
        if configs['active_channels']['fading']:
            fade_type = configs['fading_channel']['type']
            if len(configs['fading_channel']['avg_path_gains']) > 1:
                fade_effect = 'fselective'
            else:
                fade_effect = 'flat'
            fade_info = f'_{fade_type}_{fade_effect}'
            file_name += fade_info
        if configs['active_channels']['awgn']:
            file_name = f'{rx_snr}SNR_awgn' + file_name
        df = create_features(configs, rx_snr)

        file_name += f'_{qam_constel}Q_OFDM'
        save_dir = get_saving_dataset_path(configs)
        save_path = save_dir / f'{file_name}.csv'
        df.to_csv(save_path, index=False)

        if configs['superlet_scalogram']['make']:
            scalograms_dir = save_dir / 'scalograms'
            scalograms_dir.mkdir(parents=True, exist_ok=True)

            create_superlet_scalogram(df, configs['superlet_scalogram'], scalograms_dir, file_name)


def main():
    args_parser = argparse.ArgumentParser(description='Dataset synthesizer script')
    args_parser.add_argument('--dataset_config', '-d', type=str, help='Path to dataset configs',
                             default=r'/configs/dataset.json')
    args = args_parser.parse_args()

    generate_dataset(args.dataset_config)


if __name__ == '__main__':
    main()
