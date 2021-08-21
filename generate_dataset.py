import json
import argparse
import numpy as np
import pandas as pd

from src.utilities import window_pow_db, get_bayes_denoised, get_visu_denoised
from src.utilities import pow_c_a_dwt, pow_c_d_dwt, logistic_map
from src.communications_module import OFDMModulator
from src.communications_module.channels import AWGNChannel, FadingChannel
from src.utilities import get_abs_path


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
            df[signal_name] = get_visu_denoised(df['RX_OFDM'], noise)
        elif 'bayes' in feature_technique:
            df[signal_name] = get_bayes_denoised(df['RX_OFDM'], noise)
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
        file_name += f'{qam_constel}QAM'
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
            file_name += f'_{rx_snr}SNR'
            noise_model = AWGNChannel(rx_signal, rx_snr)
            rx_signal, n = noise_model.filter_x_in()
        df = create_features(configs['feature_engineering'], tx_signal, rx_signal, n)

        save_dir = get_saving_dataset_path(configs)
        save_path = save_dir / f'{file_name}.csv'

        df.to_csv(save_path, index=False)


def main():

    args_parser = argparse.ArgumentParser(description='Dataset synthesizer script')
    args_parser.add_argument('--dataset_config', '-d', type=str, help='Path to dataset configs',
                             default=r'/configs/dataset.json')
    args = args_parser.parse_args()

    generate_dataset(args.dataset_config)


if __name__ == '__main__':
    main()
