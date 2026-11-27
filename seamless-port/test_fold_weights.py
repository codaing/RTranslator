import tempfile
import unittest
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T
import onnxruntime as ort
from fold_weights import prepare

class TestFoldWeights(unittest.TestCase):
    def test_transposed_initializer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"encoder.onnx"
            weight=np.arange(12,dtype=np.float32).reshape(3,4)
            graph=h.make_graph([h.make_node("Transpose",["weight"],["rhs"],perm=[1,0]),
                h.make_node("MatMul",["x","rhs"],["y"])],"test",
                [h.make_tensor_value_info("x",T.FLOAT,[1,4])],[h.make_tensor_value_info("y",T.FLOAT,[1,3])],
                [nh.from_array(weight,name="weight")])
            model=h.make_model(graph,opset_imports=[h.make_opsetid("",17)],ir_version=9)
            onnx.save(model,str(path))
            prepared=prepare(path)
            out=onnx.load(str(prepared),load_external_data=False)
            self.assertEqual([n.op_type for n in out.graph.node],["MatMul"])
            self.assertIn("rhs",[t.name for t in out.graph.initializer])
            x=np.ones((1,4),np.float32)
            np.testing.assert_array_equal(ort.InferenceSession(str(prepared),providers=["CPUExecutionProvider"]).run(None,{"x":x})[0],x@weight.T)
