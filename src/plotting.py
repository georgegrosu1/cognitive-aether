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


def plot_roc(roc_test_df: pd.DataFrame, compare_signals, sensing_window=2, nn_model_setup: tuple = None):

    plt.figure(figsize=(15, 10))

    pfs = np.arange(0, 1, 0.01)
    gt_sgn = nn_model_setup[1]['model_cfg']['output_features'][0]
    for signal in compare_signals:
        pds = []
        sigma = estimate_sigma(roc_test_df[signal], average_sigmas=True)
        tps_num = len(np.where(roc_test_df[gt_sgn] > 0)[0])
        for fals_proba in pfs:
            positiv_cases = 0
            window = sensing_window
            for idx in range(0, len(roc_test_df[signal])):
                if len(roc_test_df[signal]) - idx <= window:
                    window -= 1
                energy = abs(roc_test_df.loc[idx:(idx+window), signal])**2
                fin_energy = np.mean(np.sum(energy))
                thresh = sigma*(invqfunc(fals_proba))*(np.sqrt(2*window)+window)
                if (fin_energy >= thresh) & (roc_test_df.loc[idx, gt_sgn] > 0):
                    positiv_cases += 1
            pd_val = positiv_cases/tps_num
            pds.append(pd_val)
        plt.plot(pfs, pds, label=signal)

    if nn_model_setup is not None:
        nn_tps, nn_fps, nn_auc_val = get_nn_roc_vals(roc_test_df=roc_test_df,
                                                     nn_model=nn_model_setup[0],
                                                     model_cfg=nn_model_setup[1])
        plt.plot(nn_fps, nn_tps, label='NN Model')

    plt.xlim(0, 1.1)
    plt.plot([0, 1], [0, 1], ls="--", c=".3")
    plt.ylabel('Probability of detection (TP)')
    plt.xlabel('Probability of false alarm (FP)')
    plt.title('Energy Detection - Reciever Operating Characteristics')
    plt.legend()
    plt.show()
