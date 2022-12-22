import os
import json
import random
import argparse
import tensorflow as tf
import numpy as np

from tensorflow_addons.metrics import F1Score
from tensorflow.keras.metrics import Recall, Precision, TruePositives, TrueNegatives, FalsePositives, FalseNegatives
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from src.models.deep_energy_detector import build_seq_model, build_resid_model, build_scalogram_model
from src.model_dev.data_preprocessing import TimeSeriesFeeder
from src.utilities import get_abs_path


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_saving_model_path(configs, model_name: str):
    save_dir = get_abs_path(configs['save_path']) / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    model_name = model_name + '_epoch{epoch:02d}_vloss{val_loss:.2f}.hdf5'
    return save_dir / model_name


def get_train_val_paths(configs):
    training_path = get_abs_path(configs['dataset']['path']) / 'train'
    validation_path = get_abs_path(configs['dataset']['path']) / 'validate'

    return training_path, validation_path


def init_training(config_path, model_name, model_type):
    abs_cfg_path = get_abs_path(config_path)
    with open(abs_cfg_path, 'r') as cfg_file:
        configs = json.load(cfg_file)

    train_cfg = configs['train_cfg']
    model_based_cfg = configs[model_type]

    if model_type == 'timeseries':
        train_timeseries_energy_detector(model_based_cfg, train_cfg, model_name)
    elif model_type == 'scalograms':
        train_scalogram_energy_detector(model_based_cfg, train_cfg, model_name)


def train_scalogram_energy_detector(model_base_cfgs, train_cfgs, model_name, model=None):
    # get training and valdiation data paths from config
    training_path, validation_path = get_train_val_paths(model_base_cfgs)

    # load training base parameters
    imgs_shape = tuple(model_base_cfgs['generator_cfg']['input_shape'])
    batch_size = train_cfgs['batch_size']
    epochs = train_cfgs['epochs']
    lr_rate = train_cfgs['lr_rate']
    pos_thresh = train_cfgs['positive_threshold']
    num_classes = len(list(training_path.glob('**/'))[1:])
    num_outputs = int(np.log2(num_classes))

    # create tf datasets from train and validation data paths
    train_ds = ImageDataGenerator()
    val_ds = ImageDataGenerator()

    train_gen = train_ds.flow_from_directory(directory=training_path,
                                             batch_size=batch_size,
                                             target_size=imgs_shape[:-1],
                                             shuffle=True)

    val_gen = val_ds.flow_from_directory(directory=validation_path,
                                         batch_size=batch_size,
                                         target_size=imgs_shape[:-1],
                                         shuffle=True)

    # prepare classification metrics
    f1_score = F1Score(num_classes=2 ** num_outputs * (num_outputs == 1) + num_outputs * (num_outputs > 1),
                       average="micro", threshold=pos_thresh)
    recall = Recall(thresholds=pos_thresh)
    precision = Precision(thresholds=pos_thresh)
    m_metrics = ['accuracy', f1_score, TruePositives(), TrueNegatives(), FalsePositives(), FalseNegatives()]

    # prepare model
    weights = model_base_cfgs['weights']
    default_cnn = model_base_cfgs['default_cnn_model']

    if model is None:
        model = build_scalogram_model(imgs_shape, num_outputs, m_metrics, default_cnn, weights, learn_rate=lr_rate)

    tfboard = tf.keras.callbacks.TensorBoard(log_dir='logdir', histogram_freq=0, write_graph=True, write_images=True)
    checkpoint_filepath = get_saving_model_path(train_cfgs, model_name)
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor=f1_score,
        mode='max',
        save_freq='epoch',
        save_best_only=False)

    model.fit(train_gen,
              epochs=epochs,
              validation_data=val_gen,
              callbacks=[model_checkpoint_callback, tfboard])


def train_timeseries_energy_detector(model_base_cfgs, train_cfgs, model_name, model=None):
    training_path, validation_path = get_train_val_paths(model_base_cfgs)

    batch_size = train_cfgs['batch_size']
    epochs = train_cfgs['epochs']
    lr_rate = train_cfgs['lr_rate']
    pos_thresh = train_cfgs['positive_threshold']

    input_features = model_base_cfgs['model_cfg']['input_features']
    output_features = model_base_cfgs['model_cfg']['output_features']
    num_outputs = len(output_features)

    window_dim = model_base_cfgs['model_cfg']['window_size']

    train_feeder = TimeSeriesFeeder(data_source=training_path,
                                    x_features=input_features,
                                    y_features=output_features,
                                    x_length=window_dim,
                                    y_length=num_outputs,
                                    batch_size=batch_size,
                                    shuffle=True)
    eval_feeder = TimeSeriesFeeder(data_source=validation_path,
                                   x_features=input_features,
                                   y_features=output_features,
                                   x_length=window_dim,
                                   y_length=num_outputs,
                                   batch_size=batch_size,
                                   shuffle=False)

    f1_score = F1Score(num_classes=2 ** num_outputs * (num_outputs == 1) + num_outputs * (num_outputs > 1),
                       average="micro", threshold=pos_thresh)
    recall = Recall(thresholds=pos_thresh)
    precision = Precision(thresholds=pos_thresh)
    m_metrics = ['accuracy', f1_score, recall, precision]

    if model is None:
        num_inputs = len(input_features)

        print("TIP:", model_base_cfgs['model_cfg']['type'])

        if 'residual' in model_base_cfgs['model_cfg']['type']:
            model = build_resid_model(input_dim=num_inputs, output_dim=num_outputs,
                                      window_dim=window_dim, custom_metrics=m_metrics,
                                      learn_rate=lr_rate)
        elif 'sequential' in model_base_cfgs['model_cfg']['type']:
            model = build_seq_model(input_dim=num_inputs, output_dim=num_outputs,
                                    window_dim=window_dim, custom_metrics=m_metrics,
                                    learn_rate=lr_rate)
    model.summary()

    tfboard = tf.keras.callbacks.TensorBoard(log_dir='logdir', histogram_freq=0, write_graph=True, write_images=True)
    checkpoint_filepath = get_saving_model_path(train_cfgs, model_name)
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor=f1_score,
        mode='max',
        save_freq='epoch',
        save_best_only=False)

    model.fit(train_feeder.feed_generator(), epochs=epochs,
              validation_data=eval_feeder.feed_generator(),
              callbacks=[model_checkpoint_callback, tfboard])


def main():

    seed_everything(seed=42)

    args_parser = argparse.ArgumentParser(description='Training script for NNs energy detection')
    args_parser.add_argument('--config_path', '-c', type=str, help='Path to config file',
                             default=r'/configs/train.json')
    args_parser.add_argument('--model_name', '-n', type=str, help='Path to model',
                             default=r'SLT_2')
    args_parser.add_argument('--model_type', '-t', type=str, help='Type of NN-based ED model (timeseries | scalograms)',
                             default=r'scalograms')
    args = args_parser.parse_args()

    init_training(args.config_path, args.model_name, args.model_type)


if __name__ == '__main__':
    main()
