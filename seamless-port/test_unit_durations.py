import tempfile
import unittest
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper as h, TensorProto as T, numpy_helper as nh
from split_unit_durations import extract,should_decode

class DurationGateTest(unittest.TestCase):
    def test_threshold_and_cursor(self):
        d=np.array([[30,19,1]],np.int64)
        self.assertTrue(should_decode(d,0,False))
        self.assertFalse(should_decode(d,1,False))
        self.assertTrue(should_decode(d,1,True))
        self.assertFalse(should_decode(d,3,True))
        with self.assertRaises(ValueError):should_decode(d,4,True)
        with self.assertRaises(ValueError):should_decode(np.array([[1,-1]]),0,True)

    def test_extract_external_graph(self):
        g=h.make_graph([
            h.make_node("Add",["chars","bias"],["durations"]),
            h.make_node("Cast",["durations"],["float_durations"],to=T.FLOAT),
            h.make_node("Mul",["float_durations","scale"],["logits"])],
            "unit_fixture",[h.make_tensor_value_info("chars",T.INT64,[1,"n"])],
            [h.make_tensor_value_info("logits",T.FLOAT,[1,"n"]),
             h.make_tensor_value_info("durations",T.INT64,[1,"n"])],
            [nh.from_array(np.array(1,np.int64),name="bias"),
             nh.from_array(np.array(2,np.float32),name="scale")])
        m=h.make_model(g,opset_imports=[h.make_opsetid("",17)],ir_version=9)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"units.onnx"
            onnx.save_model(m,str(p),save_as_external_data=True,all_tensors_to_one_file=True,
                            location="weights.bin",size_threshold=0)
            m=onnx.load(str(p),load_external_data=False)
            gate,report=extract(m)
            self.assertEqual(report["removed_nodes"],2)
            self.assertEqual([t.name for t in gate.graph.initializer],["bias"])
            q=Path(tmp)/"gate.onnx"
            onnx.save_model(gate,str(q))
            onnx.checker.check_model(str(q))
            opts=ort.SessionOptions();opts.intra_op_num_threads=1
            a=ort.InferenceSession(str(p),opts,providers=["CPUExecutionProvider"])
            b=ort.InferenceSession(str(q),opts,providers=["CPUExecutionProvider"])
            for n in (1,9):
                f={"chars":np.arange(n,dtype=np.int64)[None,:]}
                np.testing.assert_array_equal(a.run(["durations"],f)[0],b.run(None,f)[0])
