import numpy as np
import pandas as pd

from pathlib import Path
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA
from src.utilities import scale


class TimeSeriesFeeder(TimeseriesGenerator):
    def __init__(self,
                 x_features: list,
                 y_features: list,
                 x_length: int,
                 batch_size: int,
                 start_index=0,
                 end_index=None,
                 y_length=None,
                 sampling_rate=1,
                 stride: int = 1,
                 min_max_scale: bool = True,
                 pow_transform: bool = False,
                 pca_transform=False,
                 data_path: Path = None,
                 use_dataframe: pd.DataFrame = None,
                 shuffle=False,
                 reverse=False):
        self.data_path = data_path
        self.exogenous_features = x_features
        self.endogenous_features = y_features
        if y_length is None:
            self.y_length = x_length
        else:
            self.y_length = y_length
        self.min_max_scale = min_max_scale
        self.pow_transform = pow_transform
        self.pca_transform = pca_transform
        self.use_dataframe = use_dataframe

        data, targets = self._fetch_data_targets()
        super().__init__(data,
                         targets,
                         x_length,
                         sampling_rate,
                         stride,
                         start_index,
                         end_index,
                         shuffle,
                         reverse,
                         batch_size)

    # Override parent class method for data generation
    def __getitem__(self, index):
        if self.shuffle:
            rows = np.random.randint(
                self.start_index, self.end_index + 1, size=self.batch_size)
        else:
            i = self.start_index + self.batch_size * self.stride * index
            rows = np.arange(
                i, min(i + self.batch_size * self.stride, self.end_index + 1),
                self.stride)

        samples = np.array([self.data[(row - self.length):row:self.sampling_rate] for row in rows])
        targets = np.array([self.targets[row - self.y_length:row:self.sampling_rate] for row in rows])

        if self.reverse:
            return samples[:, ::-1, ...], targets
        return samples, targets

    def _fetch_data_targets(self):
        assert (self.use_dataframe is not None) | (self.data_path is not None), "Provide a dataframe or data path"
        if self.use_dataframe is None:
            df = self.get_data_from_path()
        else:
            df = self.use_dataframe
        if self.min_max_scale:
            df.loc[:, self.exogenous_features] = df.loc[:, self.exogenous_features].apply(scale, axis=0)
        if self.pow_transform:
            df.loc[:, self.exogenous_features] = self.apply_powtransform(df.loc[:, self.exogenous_features])
        if self.pca_transform:
            df.loc[:, self.exogenous_features] = self.apply_pca_transform(df.loc[:, self.exogenous_features])
        x_data = df.loc[:, self.exogenous_features].to_numpy()
        y_data = df.loc[:, self.endogenous_features].to_numpy()

        return x_data, y_data

    def apply_powtransform(self, df):
        transform_df = df.copy()
        ptransform = PowerTransformer()
        ptransform.fit(transform_df.loc[:, self.exogenous_features])
        transform_df.loc[:, self.exogenous_features] = ptransform.transform(transform_df.loc[:,
                                                                            self.exogenous_features])
        return transform_df

    def apply_pca_transform(self, df, components=None):
        if components is None:
            components = len(self.exogenous_features) % 2 + 1
        pca_t = PCA(n_components=components)
        pca_data = pca_t.fit_transform(df)
        return pca_data

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
