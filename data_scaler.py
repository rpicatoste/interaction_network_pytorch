import numpy as np
from dataclasses import dataclass

from physics_engine import POS_X, POS_Y, STATE, MASS, V_X, V_Y
from utils import listify


class DataScaler:

    def __init__(self, data):
        data = listify(data)

        self.scales = {}
        mass_stats = DataStats(data, MASS)
        x_stats = DataStats(data, POS_X)
        y_stats = DataStats(data, POS_Y)
        xy_range = np.max([x_stats.range, y_stats.range])

        self.scales[MASS] = DataScale(range=mass_stats.range, bias=0.0)
        self.scales[POS_X] = DataScale(range=xy_range, bias=x_stats.center)
        self.scales[POS_Y] = DataScale(range=xy_range, bias=y_stats.center)
        self.scales[V_X] = DataScale(range=xy_range, bias=0.0)
        self.scales[V_Y] = DataScale(range=xy_range, bias=0.0)

    def transform(self, data):
        scaled_data = np.zeros_like(data)

        for LABEL in STATE:
            scaled_data[:, LABEL, :] = self.transform_type(data[:, LABEL, :],
                                                           self.scales[LABEL])

        return scaled_data

    def transform_type(self, data, scaler):
        return (data - scaler.bias) / scaler.range

    def inverse_transform(self, scaled_data):
        data = np.zeros_like(scaled_data)

        for LABEL in STATE:
            data[:, LABEL, :] = self.inverse_transform_type(scaled_data[:, LABEL, :],
                                                            self.scales[LABEL])

        return data

    def inverse_transform_type(self, scaled_data, scaler):
        return (scaled_data * scaler.range) + scaler.bias


class DataStats:
    def __init__(self, data, label):
        self.max = np.max([np.max(sample[:, label, :]) for sample in data])
        self.min = np.min([np.min(sample[:, label, :]) for sample in data])
        self.range = self.max - self.min
        self.center = (self.max + self.min) / 2.0


@dataclass
class DataScale:
    range: float
    bias: float
