import unittest
import onnx
from onnx import helper as h,TensorProto as T
from units_quant_experiment import select_post_duration

class UnitSelectionTest(unittest.TestCase):
    def graph(self,bad=False):
        q="/decoder/layers.0/self_attn/q_proj/MatMul"
        nodes=[h.make_node("MatMul",["x","w"],["q"],name=q),
               h.make_node("MatMul",["x","w"],["early"],name="/encoder/layers.0/self_attn/q_proj/MatMul"),
               h.make_node("MatMul",["q","w"],["logits"],name="/final_proj/MatMul"),
               h.make_node("Identity",["q" if bad else "early"],["durations"])]
        return h.make_graph(nodes,"test",[],[],[h.make_tensor("w",T.FLOAT,[16,16],[0.0]*256)])
    def test_only_post_duration_attention(self):
        self.assertEqual(select_post_duration(self.graph()),["/decoder/layers.0/self_attn/q_proj/MatMul"])
    def test_reject_duration_ancestor(self):
        with self.assertRaises(RuntimeError):select_post_duration(self.graph(True))
