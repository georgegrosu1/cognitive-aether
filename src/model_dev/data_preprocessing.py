import pandas as pd

from pathlib import Path
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


class TimeSeriesFeeder:
    def __init__(self, data_path: Path, x_features: list, y_features: list,
                 window_dim: int, feed_batch: int, stride: int = 1):
        self.data_path = data_path
        self.exogenous_features = x_features
        self.endogenous_features = y_features
        self.window_dim = window_dim
        self.feed_batch = feed_batch
        self.stride = stride
        self.generator = self._init_generator()

    def _init_generator(self):
        main_dataframe = self.get_data_from_path()
        x_data = main_dataframe.loc[:, self.exogenous_features].to_numpy()
        y_data = main_dataframe.loc[:, self.endogenous_features].to_numpy()
        generator = TimeseriesGenerator(x_data, y_data, length=self.window_dim,
                                        batch_size=self.feed_batch,
                                        stride=self.stride)

        return generator

    def get_data_from_path(self):
        if 'csv' in self.data_path.suffix:
            return pd.read_csv(str(self.data_path))
        dfs = []
        all_csv_files = self.data_path.glob('*.csv')
        take_features = self.endogenous_features + self.exogenous_features
        for file_p in all_csv_files:
            df = pd.read_csv(str(file_p))
            df = df.loc[:, take_features]
            dfs.append(df)
        return pd.concat(dfs, axis=0)

    def feed_generator(self):
        return self.generator
