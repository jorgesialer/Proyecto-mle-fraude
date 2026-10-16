"""Stateless encoding for histogram boosting, with explicit missing values."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from .schema import FEATURES, TYPES

NUMERIC = tuple(name for name in FEATURES if name != 'type')
ENCODED_NAMES = (*NUMERIC, *(f'type_{value}' for value in TYPES))


class BoostingEncoder(TransformerMixin, BaseEstimator):
    """No statistics are learned; categories are defined by the source contract."""
    def fit(self, X, y=None):
        self.transform(X)
        self.n_features_in_ = len(FEATURES)
        self.feature_names_in_ = np.array(FEATURES, dtype=object)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame) or set(X.columns) != set(FEATURES):
            raise ValueError('Se requieren exclusivamente los predictores del contrato')
        if not X.type.isin(TYPES).all():
            raise ValueError('Tipo de transacción desconocido')
        result = np.empty((len(X), len(ENCODED_NAMES)), dtype=np.float32)
        result[:, :len(NUMERIC)] = X.loc[:, list(NUMERIC)].to_numpy(dtype=np.float32)
        for i, value in enumerate(TYPES):
            result[:, len(NUMERIC) + i] = X.type.eq(value).to_numpy(dtype=np.float32)
        if np.isinf(result).any():
            raise ValueError('No se admite infinito; NaN de ratio se conserva')
        return result

    def get_feature_names_out(self, input_features=None):
        return np.asarray(ENCODED_NAMES, dtype=object)


def training_weight(y) -> float:
    """A fixed policy based solely on the current training fold."""
    y = np.asarray(y)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError('El entrenamiento requiere ambas clases')
    positive = int(y.sum())
    return float(np.sqrt((len(y) - positive) / positive))
