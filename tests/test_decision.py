import unittest
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import fbeta_score
from src.fraud.decision import choose_f2_threshold,FrozenThresholdClassifier


class FixedScorer(ClassifierMixin,BaseEstimator):
    classes_=np.array([0,1])
    n_features_in_=1
    def predict_proba(self,X):
        score=np.asarray(X)[:,0]
        return np.column_stack([1-score,score])


class DecisionTests(unittest.TestCase):
    def test_threshold_matches_independent_exhaustive_f2(self):
        y=np.array([0,1,0,1,1,0,0])
        scores=np.array([.1,.2,.3,.4,.5,.6,.7])
        chosen,curve=choose_f2_threshold(y,scores)
        choices=[(fbeta_score(y,scores>=t,beta=2),t) for t in np.unique(scores)]
        best=max(choices,key=lambda pair:(pair[0],pair[1]))
        self.assertEqual(chosen,best[1])
        self.assertEqual(len(curve),len(np.unique(scores)))

    def test_equal_scores_are_included_at_the_boundary(self):
        chosen,_=choose_f2_threshold([0,1,1,0],[.5,.5,.8,.1])
        self.assertEqual(chosen,.5)
        model=FrozenThresholdClassifier(FixedScorer(),chosen)
        np.testing.assert_array_equal(model.predict([[.5],[.49],[.8]]),[1,0,1])

    def test_invalid_selection_data_is_rejected(self):
        for y,s in [([0,0],[.2,.3]),([0,1],[np.nan,.3]),([0,1],[-.1,.3]),([0,1],[.1])]:
            with self.subTest(y=y,s=s),self.assertRaises(ValueError):choose_f2_threshold(y,s)

    def test_frozen_model_does_not_refit_or_change_scores(self):
        scorer=FixedScorer();model=FrozenThresholdClassifier(scorer,.7)
        X=np.array([[.6],[.8]])
        np.testing.assert_array_equal(model.predict_proba(X),scorer.predict_proba(X))
        np.testing.assert_array_equal(model.predict(X),[0,1])
        with self.assertRaises(RuntimeError):model.fit(X,[0,1])


if __name__=='__main__':unittest.main()
