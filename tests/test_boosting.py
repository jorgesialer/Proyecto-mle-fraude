"""Contract checks for stateless encoding and training-only class weights."""
import unittest
import numpy as np
import pandas as pd
from src.fraud.boosting import BoostingEncoder, ENCODED_NAMES, NUMERIC, training_weight
from src.fraud.schema import FEATURES, TYPES


def features():
    data=pd.DataFrame({name:[0.,3.] for name in FEATURES if name!='type'})
    data['type']=['TRANSFER','PAYMENT']
    data.loc[0,'amount_to_origin_balance']=np.nan
    return data.loc[:,list(FEATURES)]


class BoostingTests(unittest.TestCase):
    def test_encoding_preserves_missing_ratio_and_contract_categories(self):
        data=features();encoded=BoostingEncoder().fit_transform(data)
        self.assertEqual(encoded.dtype,np.float32)
        self.assertEqual(encoded.shape,(2,len(ENCODED_NAMES)))
        self.assertTrue(np.isnan(encoded[0,NUMERIC.index('amount_to_origin_balance')]))
        self.assertEqual(encoded[0,len(NUMERIC)+TYPES.index('TRANSFER')],1)
        self.assertEqual(encoded[1,len(NUMERIC)+TYPES.index('PAYMENT')],1)
        np.testing.assert_array_equal(encoded[:,len(NUMERIC):].sum(axis=1),[1.,1.])

    def test_fitting_other_data_cannot_change_encodings(self):
        data=features()
        other=data.copy();other.loc[:,'amount']=1e10
        a=BoostingEncoder().fit(data).transform(data)
        b=BoostingEncoder().fit(other).transform(data)
        np.testing.assert_allclose(a,b,equal_nan=True)
        with self.assertRaises(ValueError):BoostingEncoder().transform(data.assign(isFraud=0))

    def test_weights_use_only_supplied_training_labels(self):
        self.assertEqual(training_weight([0,0,0,0,1]),2.)
        with self.assertRaises(ValueError):training_weight([0,0])

    def test_unknown_type_is_rejected_instead_of_becoming_all_zero(self):
        data=features();data.loc[0,'type']='NEW'
        with self.assertRaises(ValueError):BoostingEncoder().transform(data)


if __name__=='__main__':unittest.main()
