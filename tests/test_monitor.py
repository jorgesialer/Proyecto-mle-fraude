import unittest
import numpy as np
from src.fraud.monitor import psi,bh_reject,detect_changes


class MonitorTests(unittest.TestCase):
    def test_psi_identical_shift_and_missing(self):
        reference=np.arange(1000,dtype=float)
        self.assertAlmostEqual(psi(reference,reference),0)
        self.assertGreater(psi(reference,reference+2000),.2)
        shifted=reference.copy();shifted[:500]=np.nan
        self.assertGreater(psi(reference,shifted),.2)

    def test_unknown_category_is_counted(self):
        self.assertGreater(psi(['A']*100,['B']*100,categorical=True),.2)

    def test_bh_independent_expected_decisions(self):
        np.testing.assert_array_equal(bh_reject([.04,.001,.8,.01]),[False,True,False,True])

    def test_adwin_controlled_shift(self):
        rng=np.random.default_rng(5)
        stream=np.r_[np.zeros(5000,dtype=int),rng.binomial(1,.3,5000)]
        detected=detect_changes(stream)
        self.assertTrue(any(5000<=i<6000 for i in detected))
        self.assertFalse(any(i<5000 for i in detected))


if __name__=='__main__':unittest.main()
