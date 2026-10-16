"""Comprobar controles de inferencia y ajuste de imputación solo en training."""
import unittest
import numpy as np
import pandas as pd
from src.fraud.baseline import make_pipeline, metrics
from src.fraud.schema import FEATURES


def predictors():
    frame = pd.DataFrame({name: np.arange(1, 81, dtype=float) for name in FEATURES if name != "type"})
    frame["type"] = "TRANSFER"
    frame.loc[:9, "amount_to_origin_balance"] = np.nan
    return frame.loc[:, list(FEATURES)]


class BaselineTests(unittest.TestCase):
    def test_metadata_is_rejected_at_fit_and_predict(self):
        X = predictors()
        y = np.arange(80) % 2
        pipeline = make_pipeline("decision_tree")
        with self.assertRaises(ValueError):
            pipeline.fit(X.assign(isFraud=y), y)
        pipeline.fit(X, y)
        with self.assertRaises(ValueError):
            pipeline.predict(X.assign(step=1))

    def test_prediction_does_not_refit_training_median(self):
        X = predictors()
        y = np.arange(80) % 2
        pipeline = make_pipeline("decision_tree").fit(X, y)
        imputer = pipeline.named_steps['preprocessor'].named_transformers_['numeric']
        before = imputer.statistics_.copy()
        future = X.copy()
        future.loc[:, 'amount_to_origin_balance'] = 1e12
        pipeline.predict_proba(future)
        np.testing.assert_array_equal(imputer.statistics_, before)
        index = [name for name in FEATURES if name != 'type'].index('amount_to_origin_balance')
        self.assertEqual(imputer.statistics_[index], np.nanmedian(X.amount_to_origin_balance))

    def test_metrics_distinguish_no_alerts_and_missed_fraud(self):
        result = metrics(np.array([0, 0, 0, 1]), np.array([.1, .1, .1, .1]))
        self.assertEqual(result['average_precision'], .25)
        self.assertEqual(result['roc_auc'], .5)
        self.assertEqual(result['recall'], 0.)
        self.assertEqual(result['fn'], 1)
        self.assertEqual(result['fp'], 0)
        self.assertEqual(result['alert_rate'], 0.)

    def test_invalid_scores_and_single_class_are_rejected(self):
        with self.assertRaises(ValueError):
            metrics([0, 1], [0., np.inf])
        with self.assertRaises(ValueError):
            metrics([0, 0], [.1, .2])


if __name__ == '__main__':
    unittest.main()
