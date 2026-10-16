"""Threshold selection on a reserved validation set and a frozen decision model."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import precision_recall_curve


def choose_f2_threshold(y, scores):
    """Maximize F2 at actual unique scores; ties prefer higher threshold."""
    y=np.asarray(y);scores=np.asarray(scores)
    if y.ndim!=1 or scores.ndim!=1 or len(y)!=len(scores) or set(np.unique(y))!={0,1}:
        raise ValueError('Se requieren vectores de igual longitud y ambas clases')
    if not np.isfinite(scores).all() or ((scores<0)|(scores>1)).any():
        raise ValueError('Scores fuera de [0,1] o no finitos')
    precision,recall,thresholds=precision_recall_curve(y,scores,drop_intermediate=False)
    # The last PR point has no corresponding threshold and must be excluded.
    precision,recall=precision[:-1],recall[:-1]
    denominator=4*precision+recall
    f2=np.divide(5*precision*recall,denominator,out=np.zeros_like(denominator),where=denominator!=0)
    eligible=np.flatnonzero(np.isclose(f2,f2.max(),rtol=0,atol=1e-12))
    chosen=eligible[np.argmax(thresholds[eligible])]
    curve=pd.DataFrame({'threshold':thresholds,'precision':precision,'recall':recall,'f2':f2})
    return float(thresholds[chosen]),curve


class FrozenThresholdClassifier(ClassifierMixin, BaseEstimator):
    """A fitted scoring pipeline plus immutable-by-convention decision threshold."""
    def __init__(self,pipeline,threshold):
        self.pipeline=pipeline
        self.threshold=threshold

    @property
    def classes_(self):return self.pipeline.classes_

    @property
    def n_features_in_(self):return self.pipeline.n_features_in_

    def __sklearn_is_fitted__(self):return hasattr(self.pipeline,'classes_')

    def fit(self,X,y=None):
        raise RuntimeError('El modelo ya está fijado; no se reentrena al seleccionar umbral')

    def predict_proba(self,X):return self.pipeline.predict_proba(X)

    def predict(self,X):
        if not np.isfinite(self.threshold) or not 0<=self.threshold<=1:
            raise ValueError('Umbral fuera de [0,1]')
        if not np.array_equal(self.classes_,[0,1]):raise ValueError('Se requiere 1=fraude y 0=normal')
        return (self.predict_proba(X)[:,1]>=self.threshold).astype('int8')
