import torch
import numpy as np
from torch.utils import data

from wrt.preprocessors import Preprocessor


class NormalizingPreprocessor(Preprocessor):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        super().__init__()

    @property
    def apply_fit(self):
        return True

    @property
    def apply_predict(self):
        return True

    def __call__(self, x, y=None):
        """
        Perform data preprocessing and return preprocessed data as tuple.
        :param x: Dataset to be preprocessed.
        :param y: Labels to be preprocessed.
        :return: Preprocessed data.
        """
        mean = self.expand(self.mean, x)
        std = self.expand(self.std, x)

        x_norm = x - mean
        x_norm = x_norm / std
        x_norm = x_norm.astype(np.float32)
        return x_norm, y

    def expand(self, std, x):
        std = np.expand_dims(std, axis=0)
        std = np.expand_dims(std, axis=2)
        std = np.expand_dims(std, axis=3)
        std = np.repeat(std, repeats=x.shape[2], axis=2)
        std = np.repeat(std, repeats=x.shape[3], axis=3)

        return std

    def estimate_gradient(self, x: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        std = np.asarray(self.std, dtype=np.float32)
        gradient_back = gradient / std
        return gradient_back

    def fit(self, x, y=None, **kwargs):
        pass