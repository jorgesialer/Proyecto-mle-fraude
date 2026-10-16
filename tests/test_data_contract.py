"""Tests de límites temporales, ausencia de fuga y ratios indefinidos."""

import unittest

import numpy as np
import pandas as pd

from src.fraud.schema import engineer_features, partition_labels, temporal_folds, validate_raw


def example():
    return pd.DataFrame({
        "step": [1, 401], "type": ["TRANSFER", "PAYMENT"], "amount": [100., 50.],
        "nameOrig": ["C1", "C2"], "oldbalanceOrg": [100., 0.],
        "newbalanceOrig": [0., 0.], "nameDest": ["C3", "M1"],
        "oldbalanceDest": [0., 0.], "newbalanceDest": [100., 0.],
        "isFraud": [1, 0], "isFlaggedFraud": [0, 0],
    })


class ContractTests(unittest.TestCase):
    def test_boundaries_are_exhaustive_and_disjoint(self):
        steps = pd.Series([1, 250, 251, 300, 301, 400, 401, 744])
        self.assertEqual(partition_labels(steps).tolist(),
                         ["development"] * 2 + ["threshold_validation"] * 2 +
                         ["offline_test"] * 2 + ["production"] * 2)

    def test_invalid_steps_are_rejected(self):
        for value in [0, 745, 1.5, np.nan]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                partition_labels(pd.Series([value]))

    def test_target_existing_alert_and_ids_do_not_influence_features(self):
        raw = example()
        changed = raw.copy()
        changed["isFraud"] = 1 - raw.isFraud
        changed["isFlaggedFraud"] = 1
        changed["nameOrig"] = ["unseen1", "unseen2"]
        pd.testing.assert_frame_equal(engineer_features(raw), engineer_features(changed))
        without_labels = raw.drop(columns=["isFraud", "isFlaggedFraud"])
        pd.testing.assert_frame_equal(engineer_features(raw), engineer_features(without_labels))

    def test_zero_balance_is_not_silently_replaced_by_one(self):
        features = engineer_features(example())
        self.assertEqual(features.loc[0, "amount_to_origin_balance"], 1.)
        self.assertTrue(np.isnan(features.loc[1, "amount_to_origin_balance"]))
        self.assertEqual(features.loc[1, "origin_zero_balance"], 1)
        self.assertEqual(features.loc[1, "destination_is_merchant"], 1)

    def test_schema_rejects_nonbinary_labels(self):
        raw = example()
        validate_raw(raw)
        raw.loc[0, "isFraud"] = 2
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


if __name__ == "__main__":
    unittest.main()
