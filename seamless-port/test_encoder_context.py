import unittest
import numpy as np
from probe_encoder_context import infer_stride,splice,nrms

class EncoderContextTest(unittest.TestCase):
    def test_aligned_splice(self):
        full=np.arange(24,dtype=np.float32).reshape(1,12,2)
        prev=full[:,:10,:]
        window=full[:,4:,:].copy()
        window[:,:2,:]=-1
        np.testing.assert_array_equal(splice(prev,window,32,16,8,12),full)
        self.assertEqual(nrms(full,full),0)
    def test_reject_bad_alignment_and_length(self):
        a=np.zeros((1,10,2),np.float32)
        with self.assertRaises(ValueError):splice(a,a,3,0,8,12)
        with self.assertRaises(ValueError):splice(a,a,32,16,8,12)
        with self.assertRaises(ValueError):infer_stride(10,13,64)
        self.assertEqual(infer_stride(10,18,64),8)
