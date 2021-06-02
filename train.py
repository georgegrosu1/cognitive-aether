import os
import json
import random
import argparse
import tensorflow as tf
import numpy as np

from pathlib import Path
from tensorflow_addons.metrics import F1Score
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.callbacks import ModelCheckpoint
from src.models.deep_energy_detector import build_seq_model, build_resid_model
from src.model_dev.data_preprocessing import TimeSeriesFeeder


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_project_root() -> Path:
    return Path(__file__).absolute()


def get_abs_path(relative_path) -> Path:
    root_path = get_project_root().parent
    final_path = Path(str(root_path) + f'/{relative_path}')
    return final_path


def get_saving_model_path(configs, model_name: str):
    save_dir = get_abs_path(configs['save_path']) / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    model_name = model_name + '_epoch{epoch:02d}_vloss{val_loss:.2f}.hdf5'
    return save_dir / model_name


def get_train_val_paths(configs):
    training_path = get_abs_path(configs['dataset']['path']) / 'train'
    validation_path = get_abs_path(configs['dataset']['path']) / 'validate'

    return training_path, validation_path


def train_energy_detector(config_path, model_name, model=None):

    abs_cfg_path = get_abs_path(config_path)
    with open(abs_cfg_path, 'r') as cfg_file:
        configs = json.load(cfg_file)

    training_path, validation_path = get_train_val_paths(configs)
    batch_size = configs['train_cfg']['batch_size']
    input_features = configs['model_cfg']['input_features']
    output_features = configs['model_cfg']['output_features']
    num_outputs = configs['model_cfg']['num_outputs']
    lr_rate = configs['model_cfg']['lr_rate']
    window_dim = configs['model_cfg']['window_size']
    epochs = configs['train_cfg']['epochs']
    pos_thresh = configs['model_cfg']['positive_threshold']

    train_feeder = TimeSeriesFeeder(data_path=training_path,
                                    x_features=input_features,
                                    y_features=output_features,
                                    window_dim=window_dim,
                                    feed_batch=batch_size,
                                    shuffle=True)
    eval_feeder = TimeSeriesFeeder(data_path=validation_path,
                                   x_features=input_features,
                                   y_features=output_features,
                                   window_dim=window_dim,
                                   feed_batch=batch_size,
                                   shuffle=False)

    f1_score = F1Score(num_classes=2 ** num_outputs,
                       average="micro", threshold=pos_thresh)
    recall = Recall(thresholds=pos_thresh)
    precision = Precision(thresholds=pos_thresh)
    m_metrics = ['accuracy', f1_score, recall, precision]

    if model is None:
        num_inputs = len(input_features)

        model = build_resid_model(input_dim=num_inputs, output_dim=num_outputs,
                                  window_dim=window_dim, custom_metrics=m_metrics,
                                  learn_rate=lr_rate)

    tfboard = tf.keras.callbacks.TensorBoard(log_dir='logdir', histogram_freq=0, write_graph=True, write_images=True)
    checkpoint_filepath = get_saving_model_path(configs, model_name)
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
    args_parser.add_argument('--config_path', '-c', type=str, help='Path to model', default=r'/configs/train.json')
    args_parser.add_argument('--model_name', '-n', type=str, help='Path to model', default=r'energy_detect_v1')
    args = args_parser.parse_args()

    train_energy_detector(args.config_path, args.model_name)


if __name__ == '__main__':
    main()
