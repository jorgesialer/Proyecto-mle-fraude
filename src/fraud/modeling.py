"""Transformador con módulo estable para serialización e inferencia."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from .schema import FEATURES, TYPES


class FeatureContract(TransformerMixin, BaseEstimator):
    """Reject metadata or labels even if a downstream transformer could drop them."""
    def fit(self, X, y=None):
        self.transform(X)
        self.feature_names_in_ = np.asarray(FEATURES, dtype=object)
        self.n_features_in_ = len(FEATURES)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame) or set(X.columns) != set(FEATURES):
            raise ValueError("La inferencia requiere exactamente el contrato de predictores")
        if not X.type.isin(TYPES).all():
            raise ValueError("Tipo de transacción fuera del contrato")
        return X.loc[:, list(FEATURES)]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(FEATURES, dtype=object)
