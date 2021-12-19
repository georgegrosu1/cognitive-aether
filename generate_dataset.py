import json
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utilities import window_pow_db, get_bayes_denoised, get_visu_denoised
from src.utilities import pow_c_a_dwt, pow_c_d_dwt, logistic_map
from src.communications_module import OFDMModulator
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
                                                 continuous_transmission=ofdm_cfg['continuous'])

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

    target_signal = superlet_cfg['target_signal']
    roll_window = superlet_cfg['sliding_window']
    roll_step = superlet_cfg['step']
    gt_percent = superlet_cfg['gt_percent']
    foi = np.linspace(superlet_cfg['foi'][0], superlet_cfg['foi'][1], superlet_cfg['foi'][2])
    extent = [0, roll_window / superlet_cfg['samplerate'], foi[0], foi[-1]]
    scales = (1 / foi) / (2 * np.pi)

    start_indexes = [i for i in range(0, df.index[-1]-roll_window, roll_step)]
    end_indexes = [i for i in range(roll_window, df.index[-1], roll_step)]

    for start_idx, end_idx in zip(start_indexes, end_indexes):
        signal = df.loc[start_idx:end_idx, target_signal]
        spec = superlet(signal,
                        samplerate=superlet_cfg['samplerate'],
                        scales=scales,
                        order_max=superlet_cfg['order_max'],
                        order_min=superlet_cfg['order_min'],
                        c_1=superlet_cfg['c_1'], adaptive=superlet_cfg['adaptive'])
        ampls = np.abs(spec)

        ax.imshow(ampls, cmap="inferno", aspect="auto", extent=extent, origin='lower')

        img_name = file_name + f'_IDX_{start_idx}.png'
        gt_lookback_idx = end_idx-int(gt_percent*roll_window)
        if np.mean(df['USER'][gt_lookback_idx:end_idx]) > 0.5:
            save_path = scalograms_dir_true / img_name
        else:
            save_path = scalograms_dir_false / img_name
        fig.savefig(save_path, dpi=100, transparent=True, bbox_inches='tight', pad_inches=0)
        fig.canvas.flush_events()


def create_features(features_cfg, tx_signal, rx_signal, noise):
    df = pd.DataFrame()
    if features_cfg['denoising']:
        assert noise is not None, "Noise can not be none when applying denoising techniques." \
                                  "Please activate noise channel"

    # Create baseline features
    rolling_window = features_cfg['sliding_window_size']
    if noise is not None:
        df['noise'] = abs(noise)
        df['sigma'] = df['noise'].rolling(rolling_window).apply(np.var)
    df['TX_OFDM'] = abs(tx_signal)
    df['RX_OFDM'] = abs(rx_signal)
    df['RE_RX_OFDM'] = rx_signal.real
    df['IM_RX_OFDM'] = rx_signal.imag

    # Create denoised signals & extra features
    for feature_technique in features_cfg['denoising'] + features_cfg['extract_features']:
        upper_technique = feature_technique.upper()
        signal_name = f'RX_{upper_technique}'
        if 'visu' in feature_technique:
            df[signal_name] = get_visu_denoised(df['RX_OFDM'], df['noise'])
        elif 'bayes' in feature_technique:
            df[signal_name] = get_bayes_denoised(df['RX_OFDM'], df['noise'])
        elif 'pow_db' in feature_technique:
            df[signal_name] = df['RX_OFDM'].rolling(rolling_window).apply(window_pow_db)
        elif 'pow_logistic_map' in feature_technique:
            df[signal_name] = df['RX_BAYES'].rolling(rolling_window).apply(logistic_map)
        elif 'ca_dwt' in feature_technique:
            df[signal_name] = df['RX_OFDM'].rolling(rolling_window).apply(pow_c_a_dwt)
        elif 'cd_dwt' in feature_technique:
            df[signal_name] = df['RX_OFDM'].rolling(rolling_window).apply(pow_c_d_dwt)

    # Remove possible residual NaNs from end of dataframe
    last_nan = np.where(np.asanyarray(np.isnan(df)))[0][-1]
    df = df.loc[(last_nan + 1):, :]

    # Generate Ground Truth based on TX signal
    ones = df['TX_OFDM'] > 0
    df.loc[ones, 'USER'] = 1
    df.loc[~ones, 'USER'] = 0

    return df


def generate_dataset(dataset_cfg):

    abs_cfg_path = get_abs_path(dataset_cfg)
    with open(abs_cfg_path, 'r') as cfg_file:
        configs = json.load(cfg_file)

    n = None
    for rx_snr in configs['awgn_channel']['rx_snrs_list']:
        file_name = f''
        tx_signal = get_ofdm_data(configs['ofdm_moodulator'])
        qam_constel = 2**int(configs['ofdm_moodulator']['bits_per_sym'])
        rx_signal = tx_signal
        if configs['active_channels']['fading']:
            rx_signal = get_channel_faded_data(configs['fading_channel'], tx_signal)
            fade_effect = 'flat'
            fade_type = configs['fading_channel']['type']
            if len(configs['fading_channel']['avg_path_gains']) > 1:
                fade_effect = 'fselective'
            fade_info = f'_{fade_type}_{fade_effect}'
            file_name += fade_info
        if configs['active_channels']['awgn']:
            file_name = f'{rx_snr}SNR_awgn' + file_name
            noise_model = AWGNChannel(rx_signal, rx_snr)
            rx_signal, n = noise_model.filter_x_in()
        df = create_features(configs['feature_engineering'], tx_signal, rx_signal, n)

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
