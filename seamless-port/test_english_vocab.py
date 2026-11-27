import unittest
import numpy as np
from onnx import helper as h,numpy_helper as nh,TensorProto as T
from onnx.reference import ReferenceEvaluator
from prune_english_vocab import transform
from select_english_vocab import select

class EnglishVocabularyTest(unittest.TestCase):
    def test_selection_retains_required_and_symbols(self):
        p=["<pad>","<unk>","▁hello","你好","∫","1","▁world","α"]
        k=select(p,{0,1},7)
        self.assertTrue({0,1,4,5,7}.issubset(k));self.assertNotIn(3,k)
    def test_tied_projection_preserves_original_logits(self):
        w=np.arange(12,dtype=np.float32).reshape(6,2)/10
        g=h.make_graph([h.make_node("Gather",["w","tokens"],["features"]),
             h.make_node("Transpose",["w"],["wt"],perm=[1,0]),
             h.make_node("MatMul",["features","wt"],["logits"])],
             "tiny",[h.make_tensor_value_info("tokens",T.INT64,[1,1])],
             [h.make_tensor_value_info("logits",T.FLOAT,[1,1,6])],
             [nh.from_array(w,name="w")])
        m=h.make_model(g,opset_imports=[h.make_opsetid("",17)],ir_version=9)
        inputs={"tokens":np.array([[4]],np.int64)}
        ref=ReferenceEvaluator(m).run(None,inputs)[0]
        kept=[1,3,4]
        self.assertEqual(transform(m,kept,6,1),{"w":0})
        m.graph.initializer[0].CopyFrom(nh.from_array(w[kept],name="w"))
        out=ReferenceEvaluator(m).run(None,inputs)[0]
        np.testing.assert_allclose(out[...,kept],ref[...,kept])
        self.assertTrue(np.all(out[...,[0,2,5]]==np.finfo(np.float32).min))
