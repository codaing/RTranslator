import unittest
import numpy as np
from prefill_reuse import retained_prefix,retain_state

class PrefillReuseTest(unittest.TestCase):
    def test_prefix_selection(self):
        ids=[2,7,8,9,10]
        self.assertEqual(retained_prefix(ids,tuple(ids),1,False),4)
        self.assertEqual(retained_prefix(ids,tuple(ids),8,False),0)
        self.assertEqual(retained_prefix(ids,tuple(ids),1,True),0)
        self.assertEqual(retained_prefix(ids,None,1,False),0)
        with self.assertRaises(ValueError):retained_prefix(ids,[2,7,8],1,False)
    def test_cache_layout(self):
        cache=np.arange(30,dtype=np.float32).reshape(1,2,5,3)
        features=[np.array([i],np.float32) for i in range(5)]
        retained,rows=retain_state([cache,cache],features,3)
        self.assertEqual(len(rows),3)
        np.testing.assert_array_equal(retained[0],cache[:,:,:3,:])
        self.assertTrue(retained[0].flags.c_contiguous)
        with self.assertRaises(ValueError):retain_state([cache],features[:-1],3)
