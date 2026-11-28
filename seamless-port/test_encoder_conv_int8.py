import tempfile
from pathlib import Path
import unittest
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper as h, numpy_helper as nh, TensorProto as T
from encoder_conv_int8 import selection,PrefixReader,calibrate

def graph():
    # Encoder-like [B,T,80] input, pointwise then grouped Conv.
    rng=np.random.default_rng(42)
    w=rng.normal(0,.08,(4,80,1)).astype(np.float32)
    d=rng.normal(0,.1,(4,1,3)).astype(np.float32)
    ns=[h.make_node("Transpose",["fbank"],["x"],perm=[0,2,1]),
        h.make_node("Conv",["x","w"],["y"],name="point"),
        h.make_node("Conv",["y","d"],["encoded"],name="depth",group=4,pads=[1,1])]
    return h.make_model(h.make_graph(ns,"encoder",
        [h.make_tensor_value_info("fbank",T.FLOAT,[1,None,80])],
        [h.make_tensor_value_info("encoded",T.FLOAT,[1,4,None])],
        [nh.from_array(w,name="w"),nh.from_array(d,name="d")]),
        opset_imports=[h.make_opsetid("",17)],ir_version=9)

class ConvInt8Test(unittest.TestCase):
    def test_reader(self):
        r=PrefixReader(np.zeros((256,80),np.float32),[192,256])
        self.assertEqual(r.get_next()["fbank"].shape,(1,192,80))
        self.assertEqual(r.get_next()["fbank"].shape,(1,256,80))
        self.assertIsNone(r.get_next())
        r.rewind();self.assertEqual(r.get_next()["fbank"].shape,(1,192,80))
    def test_inventory(self):
        self.assertEqual(len(selection(graph(),"all")),2)
        self.assertEqual([x["name"] for x in selection(graph(),"pointwise")],["point"])
    def test_quantize_both_formats(self):
        features=np.random.default_rng(43).normal(size=(320,80)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/"original.onnx"
            onnx.save_model(graph(),str(source))
            opt=ort.SessionOptions();opt.intra_op_num_threads=1
            feeds={"fbank":features[None,:256]}
            expected=ort.InferenceSession(str(source),opt,providers=["CPUExecutionProvider"]).run(None,feeds)[0]
            for profile,count in (("qdq-all",2),("qoperator-all",2),("qoperator-pointwise",1)):
                target=root/(profile+".onnx")
                report=calibrate(source,target,features,profile,2)
                self.assertEqual(report["selected"],count)
                result=ort.InferenceSession(str(target),opt,providers=["CPUExecutionProvider"]).run(None,feeds)[0]
                self.assertEqual(result.shape,expected.shape)
                self.assertTrue(np.isfinite(result).all())
                self.assertLess(np.linalg.norm(result-expected)/np.linalg.norm(expected),.12)
if __name__=="__main__":unittest.main()
