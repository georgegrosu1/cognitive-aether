import pandas as pd

from pathlib import Path
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import PowerTransformer
from src.utilities import scale


class TimeSeriesFeeder:
    def __init__(self, data_path: Path, x_features: list, y_features: list,
                 window_dim: int, feed_batch: int, stride: int = 1,
                 min_max_scale: bool = True, pow_transform: bool = True):
        self.data_path = data_path
        self.exogenous_features = x_features
        self.endogenous_features = y_features
        self.window_dim = window_dim
        self.feed_batch = feed_batch
        self.stride = stride
        self.min_max_scale = min_max_scale
        self.pow_transform = pow_transform
        self.generator = self._init_generator()

    def _init_generator(self):
        main_dataframe = self.get_data_from_path()
        x_data = main_dataframe.loc[:, self.exogenous_features].to_numpy()
        y_data = main_dataframe.loc[:, self.endogenous_features].to_numpy()
        generator = TimeseriesGenerator(x_data, y_data, length=self.window_dim,
                                        batch_size=self.feed_batch,
                                        stride=self.stride, shuffle=True)

        return generator

    def apply_powtransform(self, df):
        transform_df = df.copy()
        ptransform = PowerTransformer()
        ptransform.fit(transform_df.loc[:, self.exogenous_features])
        transform_df.loc[:, self.exogenous_features] = ptransform.transform(transform_df.loc[:,
                                                                            self.exogenous_features])
        return transform_df

    def get_data_from_path(self):
        if 'csv' in self.data_path.suffix:
            return pd.read_csv(str(self.data_path))
        dfs = []
        all_csv_files = self.data_path.glob('*.csv')
        take_features = self.endogenous_features + self.exogenous_features

        for file_p in all_csv_files:
            df = pd.read_csv(str(file_p))
            df = df.loc[:len(df)/3, take_features]
            if self.min_max_scale:
                df = df.apply(scale, axis=0)
            if self.pow_transform:
                df = self.apply_powtransform(df)
            dfs.append(df)
        return pd.concat(dfs, axis=0)

    def feed_generator(self):
        return self.generator
