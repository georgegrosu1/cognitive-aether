import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.utilities import invqfunc
from sklearn.metrics import roc_curve, auc
from src.model_dev.data_preprocessing import TimeSeriesFeeder
from skimage.restoration import estimate_sigma


def get_nn_roc_vals(roc_test_df: pd.DataFrame, nn_model, model_cfg):

    input_features = model_cfg['model_cfg']['input_features']
    output_features = model_cfg['model_cfg']['output_features']
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


def plot_roc(roc_test_df: pd.DataFrame, compare_signals, json_cfg, sensing_window=2, nn_model=None):

    plt.figure(figsize=(15, 10))

    pfs = np.arange(0, 1, 0.01)
    gt_sgn = json_cfg['model_cfg']['output_features'][0]
    gt_tps_num = len(np.where(roc_test_df.loc[(sensing_window-1):, gt_sgn] > 0)[0])
    gt_tns_num = len(np.where(roc_test_df.loc[(sensing_window-1):, gt_sgn] == 0)[0])
    window = sensing_window
    for signal in compare_signals:
        pds_val, pfs_val = [], []
        for fals_proba in pfs:
            true_positive_cases = 0
            false_positive_cases = 0
            for idx in range(window, len(roc_test_df)):
                slide_window_data = roc_test_df.loc[(idx-window):idx, signal].values
                slide_window_sigma = roc_test_df.loc[(idx - window):idx, 'sigma'].values
                sigma = estimate_sigma(slide_window_sigma, average_sigmas=True)
                energy = np.mean(abs(slide_window_data**2))
                thresh = sigma*(invqfunc(fals_proba)) * (np.sqrt(2*window) + window)
                if energy >= thresh:
                    if roc_test_df.loc[idx, gt_sgn] > 0:
                        true_positive_cases += 1
                    else:
                        false_positive_cases += 1
            pd_val = true_positive_cases/gt_tps_num
            pf_val = false_positive_cases/gt_tns_num
            pds_val.append(pd_val)
            pfs_val.append(pf_val)
        plt.plot(pfs, pds_val, label=signal)

    if nn_model is not None:
        nn_tps, nn_fps, nn_auc_val = get_nn_roc_vals(roc_test_df=roc_test_df,
                                                     nn_model=nn_model,
                                                     model_cfg=json_cfg)
        plt.plot(nn_fps, nn_tps, label='NN Model')

    plt.xlim(0, 1.1)
    plt.plot([0, 1], [0, 1], ls="--", c=".3")
    plt.ylabel('Probability of detection (TP)')
    plt.xlabel('Probability of false alarm (FP)')
    plt.title('Energy Detection - Reciever Operating Characteristics')
    plt.legend()
    plt.show()
