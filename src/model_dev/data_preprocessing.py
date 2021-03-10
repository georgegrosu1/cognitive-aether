import pandas as pd

from pathlib import Path
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


class TimeSeriesFeeder:
    def __init__(self, data_path: Path, target_features: list, window_dim: int, feed_batch: int):
        self.data_path = data_path
        self.target_features = target_features
        self.window_dim = window_dim
        self.main_dataframe = self.get_data_from_path()
        self.feed_batch = feed_batch

    def get_data_from_path(self):
        if 'csv' in self.data_path.suffix:
            return pd.read_csv(str(self.data_path))
        dfs = []
        for file_p in self.data_path.glob('*.csv'):
            df = pd.read_csv(str(file_p))
            dfs.append(df)
        return pd.concat(dfs, axis=0)

    def feed_generator(self, batch_dim=None):
        data = self.main_dataframe.to_numpy()
        targets = self.main_dataframe.index.tolist()
        if batch_dim is not None:
            batch_s = batch_dim
        else:
            batch_s = self.feed_batch
        generator = TimeseriesGenerator(data, targets, length=self.window_dim, batch_size=batch_s)
        return generator
