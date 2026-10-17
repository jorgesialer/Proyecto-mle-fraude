"""Contract, encoding, decision and monitoring checks, grouped by test class."""
import unittest
import numpy as np
import pandas as pd
from src.fraud.baseline import make_pipeline, metrics
from src.fraud.schema import FEATURES

def predictors():
    frame = pd.DataFrame({name: np.arange(1, 81, dtype=float) for name in FEATURES if name != 'type'})
    frame['type'] = 'TRANSFER'
    frame.loc[:9, 'amount_to_origin_balance'] = np.nan
    return frame.loc[:, list(FEATURES)]

class BaselineTests(unittest.TestCase):

    def test_metadata_is_rejected_at_fit_and_predict(self):
        X = predictors()
        y = np.arange(80) % 2
        pipeline = make_pipeline('decision_tree')
        with self.assertRaises(ValueError):
            pipeline.fit(X.assign(isFraud=y), y)
        pipeline.fit(X, y)
        with self.assertRaises(ValueError):
            pipeline.predict(X.assign(step=1))

    def test_prediction_does_not_refit_training_median(self):
        X = predictors()
        y = np.arange(80) % 2
        pipeline = make_pipeline('decision_tree').fit(X, y)
        imputer = pipeline.named_steps['preprocessor'].named_transformers_['numeric']
        before = imputer.statistics_.copy()
        future = X.copy()
        future.loc[:, 'amount_to_origin_balance'] = 1000000000000.0
        pipeline.predict_proba(future)
        np.testing.assert_array_equal(imputer.statistics_, before)
        index = [name for name in FEATURES if name != 'type'].index('amount_to_origin_balance')
        self.assertEqual(imputer.statistics_[index], np.nanmedian(X.amount_to_origin_balance))

    def test_metrics_distinguish_no_alerts_and_missed_fraud(self):
        result = metrics(np.array([0, 0, 0, 1]), np.array([0.1, 0.1, 0.1, 0.1]))
        self.assertEqual(result['average_precision'], 0.25)
        self.assertEqual(result['roc_auc'], 0.5)
        self.assertEqual(result['recall'], 0.0)
        self.assertEqual(result['fn'], 1)
        self.assertEqual(result['fp'], 0)
        self.assertEqual(result['alert_rate'], 0.0)

    def test_invalid_scores_and_single_class_are_rejected(self):
        with self.assertRaises(ValueError):
            metrics([0, 1], [0.0, np.inf])
        with self.assertRaises(ValueError):
            metrics([0, 0], [0.1, 0.2])
import unittest
import numpy as np
import pandas as pd
from src.fraud.boosting import BoostingEncoder, ENCODED_NAMES, NUMERIC, training_weight
from src.fraud.schema import FEATURES, TYPES

def features():
    data = pd.DataFrame({name: [0.0, 3.0] for name in FEATURES if name != 'type'})
    data['type'] = ['TRANSFER', 'PAYMENT']
    data.loc[0, 'amount_to_origin_balance'] = np.nan
    return data.loc[:, list(FEATURES)]

class BoostingTests(unittest.TestCase):

    def test_encoding_preserves_missing_ratio_and_contract_categories(self):
        data = features()
        encoded = BoostingEncoder().fit_transform(data)
        self.assertEqual(encoded.dtype, np.float32)
        self.assertEqual(encoded.shape, (2, len(ENCODED_NAMES)))
        self.assertTrue(np.isnan(encoded[0, NUMERIC.index('amount_to_origin_balance')]))
        self.assertEqual(encoded[0, len(NUMERIC) + TYPES.index('TRANSFER')], 1)
        self.assertEqual(encoded[1, len(NUMERIC) + TYPES.index('PAYMENT')], 1)
        np.testing.assert_array_equal(encoded[:, len(NUMERIC):].sum(axis=1), [1.0, 1.0])

    def test_fitting_other_data_cannot_change_encodings(self):
        data = features()
        other = data.copy()
        other.loc[:, 'amount'] = 10000000000.0
        a = BoostingEncoder().fit(data).transform(data)
        b = BoostingEncoder().fit(other).transform(data)
        np.testing.assert_allclose(a, b, equal_nan=True)
        with self.assertRaises(ValueError):
            BoostingEncoder().transform(data.assign(isFraud=0))

    def test_weights_use_only_supplied_training_labels(self):
        self.assertEqual(training_weight([0, 0, 0, 0, 1]), 2.0)
        with self.assertRaises(ValueError):
            training_weight([0, 0])

    def test_unknown_type_is_rejected_instead_of_becoming_all_zero(self):
        data = features()
        data.loc[0, 'type'] = 'NEW'
        with self.assertRaises(ValueError):
            BoostingEncoder().transform(data)
import unittest
import numpy as np
import pandas as pd
from src.fraud.schema import engineer_features, partition_labels, temporal_folds, validate_raw

def example():
    return pd.DataFrame({'step': [1, 401], 'type': ['TRANSFER', 'PAYMENT'], 'amount': [100.0, 50.0], 'nameOrig': ['C1', 'C2'], 'oldbalanceOrg': [100.0, 0.0], 'newbalanceOrig': [0.0, 0.0], 'nameDest': ['C3', 'M1'], 'oldbalanceDest': [0.0, 0.0], 'newbalanceDest': [100.0, 0.0], 'isFraud': [1, 0], 'isFlaggedFraud': [0, 0]})

class ContractTests(unittest.TestCase):

    def test_boundaries_are_exhaustive_and_disjoint(self):
        steps = pd.Series([1, 250, 251, 300, 301, 400, 401, 744])
        self.assertEqual(partition_labels(steps).tolist(), ['development'] * 2 + ['threshold_validation'] * 2 + ['offline_test'] * 2 + ['production'] * 2)

    def test_invalid_steps_are_rejected(self):
        for value in [0, 745, 1.5, np.nan]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                partition_labels(pd.Series([value]))

    def test_target_existing_alert_and_ids_do_not_influence_features(self):
        raw = example()
        changed = raw.copy()
        changed['isFraud'] = 1 - raw.isFraud
        changed['isFlaggedFraud'] = 1
        changed['nameOrig'] = ['unseen1', 'unseen2']
        pd.testing.assert_frame_equal(engineer_features(raw), engineer_features(changed))
        without_labels = raw.drop(columns=['isFraud', 'isFlaggedFraud'])
        pd.testing.assert_frame_equal(engineer_features(raw), engineer_features(without_labels))

    def test_zero_balance_is_not_silently_replaced_by_one(self):
        features = engineer_features(example())
        self.assertEqual(features.loc[0, 'amount_to_origin_balance'], 1.0)
        self.assertTrue(np.isnan(features.loc[1, 'amount_to_origin_balance']))
        self.assertEqual(features.loc[1, 'origin_zero_balance'], 1)
        self.assertEqual(features.loc[1, 'destination_is_merchant'], 1)

    def test_schema_rejects_nonbinary_labels(self):
        raw = example()
        validate_raw(raw)
        raw.loc[0, 'isFraud'] = 2
        with self.assertRaises(ValueError):
            validate_raw(raw)

    def test_folds_keep_whole_hours_and_respect_causality(self):
        steps = pd.Series(np.repeat(np.arange(1, 251), 3))
        for train, valid in temporal_folds(steps):
            self.assertLess(steps.iloc[train].max(), steps.iloc[valid].min())
            self.assertFalse(set(steps.iloc[train]) & set(steps.iloc[valid]))
            self.assertEqual(len(train) % 3, 0)
            self.assertEqual(len(valid) % 3, 0)
        with self.assertRaises(ValueError):
            list(temporal_folds(pd.Series([1, 401])))
import unittest
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import fbeta_score
from src.fraud.decision import choose_f2_threshold, FrozenThresholdClassifier

class FixedScorer(ClassifierMixin, BaseEstimator):
    classes_ = np.array([0, 1])
    n_features_in_ = 1

    def predict_proba(self, X):
        score = np.asarray(X)[:, 0]
        return np.column_stack([1 - score, score])

class DecisionTests(unittest.TestCase):

    def test_threshold_matches_independent_exhaustive_f2(self):
        y = np.array([0, 1, 0, 1, 1, 0, 0])
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        chosen, curve = choose_f2_threshold(y, scores)
        choices = [(fbeta_score(y, scores >= t, beta=2), t) for t in np.unique(scores)]
        best = max(choices, key=lambda pair: (pair[0], pair[1]))
        self.assertEqual(chosen, best[1])
        self.assertEqual(len(curve), len(np.unique(scores)))

    def test_equal_scores_are_included_at_the_boundary(self):
        chosen, _ = choose_f2_threshold([0, 1, 1, 0], [0.5, 0.5, 0.8, 0.1])
        self.assertEqual(chosen, 0.5)
        model = FrozenThresholdClassifier(FixedScorer(), chosen)
        np.testing.assert_array_equal(model.predict([[0.5], [0.49], [0.8]]), [1, 0, 1])

    def test_invalid_selection_data_is_rejected(self):
        for y, s in [([0, 0], [0.2, 0.3]), ([0, 1], [np.nan, 0.3]), ([0, 1], [-0.1, 0.3]), ([0, 1], [0.1])]:
            with self.subTest(y=y, s=s), self.assertRaises(ValueError):
                choose_f2_threshold(y, s)

    def test_frozen_model_does_not_refit_or_change_scores(self):
        scorer = FixedScorer()
        model = FrozenThresholdClassifier(scorer, 0.7)
        X = np.array([[0.6], [0.8]])
        np.testing.assert_array_equal(model.predict_proba(X), scorer.predict_proba(X))
        np.testing.assert_array_equal(model.predict(X), [0, 1])
        with self.assertRaises(RuntimeError):
            model.fit(X, [0, 1])
import unittest
import numpy as np
from src.fraud.monitor import psi, bh_reject, detect_changes

class MonitorTests(unittest.TestCase):

    def test_psi_identical_shift_and_missing(self):
        reference = np.arange(1000, dtype=float)
        self.assertAlmostEqual(psi(reference, reference), 0)
        self.assertGreater(psi(reference, reference + 2000), 0.2)
        shifted = reference.copy()
        shifted[:500] = np.nan
        self.assertGreater(psi(reference, shifted), 0.2)

    def test_unknown_category_is_counted(self):
        self.assertGreater(psi(['A'] * 100, ['B'] * 100, categorical=True), 0.2)

    def test_bh_independent_expected_decisions(self):
        np.testing.assert_array_equal(bh_reject([0.04, 0.001, 0.8, 0.01]), [False, True, False, True])

    def test_adwin_controlled_shift(self):
        rng = np.random.default_rng(5)
        stream = np.r_[np.zeros(5000, dtype=int), rng.binomial(1, 0.3, 5000)]
        detected = detect_changes(stream)
        self.assertTrue(any((5000 <= i < 6000 for i in detected)))
        self.assertFalse(any((i < 5000 for i in detected)))
