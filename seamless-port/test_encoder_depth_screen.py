import unittest
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper as h, numpy_helper as nh, TensorProto as T
from encoder_depth_screen import rewrite,PROFILES

def graph(hidden=False):
    nodes=[];weights=[];previous="fbank"
    for i in range(4):
        prefix=f"/model/speech_encoder/inner/layers.{i}/"
        weight=f"w{i}"
        weights.append(nh.from_array(np.eye(4,dtype=np.float32)*(i+1),name=weight))
        nodes.extend([h.make_node("MatMul",[previous,weight],[prefix+"a"],name=prefix+"MatMul"),
                      h.make_node("Add",[previous,prefix+"a"],[prefix+"b"],name=prefix+"Add")])
        previous=prefix+"b"
    if hidden:
        nodes.append(h.make_node("Add",[previous,"/model/speech_encoder/inner/layers.1/a"],["encoded"],name="out"))
    else:nodes.append(h.make_node("Identity",[previous],["encoded"],name="out"))
    return h.make_model(h.make_graph(nodes,"test",[h.make_tensor_value_info("fbank",T.FLOAT,[1,None,4])],
           [h.make_tensor_value_info("encoded",T.FLOAT,[1,None,4])],weights),opset_imports=[h.make_opsetid("",17)],ir_version=9)

class DepthTest(unittest.TestCase):
    def test_bypass_consecutive_blocks(self):
        model=graph()
        report=rewrite(model,(0,3),4)
        self.assertEqual(report["removed"],[1,2])
        self.assertEqual(len(model.graph.initializer),2)
        onnx.checker.check_model(model)
        x=np.arange(12,dtype=np.float32).reshape(1,3,4)
        opts=ort.SessionOptions();opts.intra_op_num_threads=1
        a=ort.InferenceSession(model.SerializeToString(),opts,providers=["CPUExecutionProvider"]).run(None,{"fbank":x})[0]
        np.testing.assert_array_equal(a,x*10)
    def test_hidden_dependency_rejected(self):
        with self.assertRaises(ValueError):rewrite(graph(True),(0,3),4)
    def test_profile_inventory(self):
        self.assertEqual([len(v) for v in PROFILES.values()],[18,12,6])
if __name__=="__main__":unittest.main()
