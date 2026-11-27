import unittest
import numpy as np
import onnxruntime as ort
from onnx import helper as h,numpy_helper as nh,TensorProto as T
from quantize_tied_vocab import rewrite,quantize_rows

class TiedVocabularyTest(unittest.TestCase):
    def test_native_projection_and_row_gather(self):
        rng=np.random.default_rng(42)
        w=rng.normal(size=(32,64)).astype(np.float32)
        g=h.make_graph([h.make_node("Gather",["w","tokens"],["features"]),
            h.make_node("Transpose",["w"],["wt"],perm=[1,0]),
            h.make_node("MatMul",["features","wt"],["projection_output"]),
            h.make_node("Cast",["projection_output"],["logits"],name="projection_output",to=T.FLOAT)],
            "tiny",[h.make_tensor_value_info("tokens",T.INT64,[1,None])],
            [h.make_tensor_value_info("features",T.FLOAT,[1,None,64]),
             h.make_tensor_value_info("logits",T.FLOAT,[1,None,32])],
            [nh.from_array(w,name="w")])
        m=h.make_model(g,opset_imports=[h.make_opsetid("",17)],ir_version=9)
        rewrite(m,32,64)
        names=[n.name for n in m.graph.node if n.name]
        self.assertEqual(len(names),len(set(names)))
        q,s=quantize_rows(w,64)
        m.graph.initializer.extend([nh.from_array(q,name="tied8_q"),
                                   nh.from_array(s.reshape(-1),name="tied8_scales")])
        session=ort.InferenceSession(m.SerializeToString(),providers=["CPUExecutionProvider"])
        tokens=np.array([[1,7,31]],np.int64)
        features,logits=session.run(None,{"tokens":tokens})
        restored=((q.astype(np.float32)-128)*s[...,None]).reshape(w.shape)
        np.testing.assert_allclose(features,restored[tokens],rtol=1e-6,atol=1e-6)
        np.testing.assert_allclose(logits,restored[tokens]@restored.T,rtol=1e-4,atol=1e-4)
        expected=w[tokens]@w.T
        self.assertLess(np.linalg.norm(logits-expected)/np.linalg.norm(expected),.02)
        self.assertTrue(np.array_equal(logits.argmax(-1),expected.argmax(-1)))
