import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.models.dynamic_th_ed_algo import EDDector
from sklearn.metrics import roc_curve, auc
from src.model_dev.data_preprocessing import TimeSeriesFeeder


def get_nn_roc_vals(sgn, roc_test_df: pd.DataFrame, nn_model, model_cfg):

    snr_v = sgn.split('_')[0]
    addit = [f'{snr_v}_{outp}' for outp in model_cfg['model_cfg']['input_features'][1:]]
    input_features = [sgn] + addit
    output_features = [f'{snr_v}_{outp}' for outp in model_cfg['model_cfg']['output_features']]
    window_dim = model_cfg['model_cfg']['window_size']

    test_feeder = TimeSeriesFeeder(use_dataframe=roc_test_df,
                                   x_features=input_features,
                                   y_features=output_features,
                                   window_dim=window_dim,
                                   feed_batch=1)

    y_pred = nn_model.predict(test_feeder.feed_generator()).ravel()

    y_test = roc_test_df.loc[:, output_features]
    print(y_pred.shape, y_test[:-window_dim].shape)
    nn_fpr, nn_tpr, nn_thresholds = roc_curve(y_test[:-window_dim], y_pred)
    auc_keras = auc(nn_fpr, nn_tpr)

    return nn_tpr, nn_fpr, auc_keras


def plot_roc(roc_test_df: pd.DataFrame, compare_signals, json_cfg, sensing_window=2, nn_model=()):
    linestyle_tuple = [
        ('loosely dotted', (0, (1, 10))),
        ('dotted', (0, (1, 1))),
        ('densely dotted', (0, (1, 1))),

        ('loosely dashed', (0, (5, 10))),
        ('dashed', (0, (5, 5))),
        ('densely dashed', (0, (5, 1))),

        ('loosely dashdotted', (0, (3, 10, 1, 10))),
        ('dashdotted', (0, (3, 5, 1, 5))),
        ('densely dashdotted', (0, (3, 1, 1, 1))),

        ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
        ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
        ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

    plt.figure(figsize=(15, 10))
    pfs = np.arange(0, 1.02, 0.02)

    window = sensing_window
    cnt = 0
    ed_detector = EDDector(window)
    for idj, signal in enumerate(compare_signals):
        snr_v = signal.split('_')[0]
        user = json_cfg['model_cfg']['output_features'][0]
        gt_sgn = f'{snr_v}_{user}'
        gt_tps_num = len(np.where(roc_test_df.loc[(sensing_window - 1):, gt_sgn] > 0)[0])
        print(gt_tps_num, gt_sgn)
        gt_tns_num = len(np.where(roc_test_df.loc[(sensing_window - 1):, gt_sgn] == 0)[0])
        pds_val, pfs_val = [], []
        for fals_proba in pfs:
            true_positive_cases = 0
            false_positive_cases = 0
            for idx in range(window, len(roc_test_df)):
                slide_window_data = roc_test_df.loc[(idx-window):idx, signal].values
                slide_window_noise = roc_test_df.loc[(idx - window):idx, f'{snr_v}_noise'].values
                # sigma = estimate_sigma(slide_window_noise, average_sigmas=True)**2
                ed_pred = ed_detector.predict(slide_window_data, slide_window_noise, fals_proba)
                # This is a TP case
                if roc_test_df.loc[idx, gt_sgn] == 1 and ed_pred == 1:
                    true_positive_cases += 1
                # This is a FP case
                elif roc_test_df.loc[idx, gt_sgn] != 1 and ed_pred == 1:
                    false_positive_cases += 1
            pd_val = true_positive_cases/gt_tps_num
            pf_val = false_positive_cases/gt_tns_num
            pds_val.append(pd_val)
            pfs_val.append(pf_val)

        denoi = signal.split('_')[-1]
        ed_auc = auc(pfs, pds_val)
        plt.plot(pfs, pds_val, label=f'{denoi} ED @ {snr_v} | AUC: {ed_auc:.3f}',
                 linestyle=linestyle_tuple[idj][1], linewidth=1.5)
        cnt = idj

    use_sgn = [sgn for sgn in compare_signals if 'BAYES' in sgn]
    for model_name, nned_model in nn_model:
        for idj, signal in enumerate(use_sgn):
            snr_v = signal.split('_')[0]
            nn_tps, nn_fps, nn_auc_val = get_nn_roc_vals(signal,
                                                         roc_test_df=roc_test_df,
                                                         nn_model=nned_model,
                                                         model_cfg=json_cfg)
            print(snr_v, model_name, nn_auc_val)
            plt.plot(nn_fps, nn_tps, label=f'NN {model_name} @ {snr_v} | AUC: {nn_auc_val:.3f}',
                     linestyle=linestyle_tuple[(idj+cnt)][1], linewidth=2.5)

    plt.xlim(0, 1.1)
    plt.plot([0, 1], [0, 1], ls="--", c=".3")
    plt.ylabel('Probability of detection (TP)')
    plt.xlabel('Probability of false alarm (FP)')
    plt.title('Energy Detection - Reciever Operating Characteristics')
    plt.legend()
    plt.show()
