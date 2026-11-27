import tempfile
import unittest
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper as h,numpy_helper as nh,TensorProto as T
from weight_only_int8 import convert,grouped_weight

class TestWeightOnly(unittest.TestCase):
    def test_grouped_zeros_and_outliers(self):
        w=np.zeros((128,2),np.float32);w[0,0]=120;w[64:,0]=0.125
        q,s=grouped_weight(w,group=64)
        reconstructed=(q.astype(np.float32)*s[:,None]).reshape(2,128).T
        self.assertTrue(np.isfinite(reconstructed).all())
        np.testing.assert_allclose(reconstructed,w,atol=0.001)

    def test_finer_groups_preserve_small_weights_near_outliers(self):
        w=np.full((64,2),0.125,np.float32)
        w[0,:]=100
        def reconstruct(group):
            q,s=grouped_weight(w,group=group)
            return (q.astype(np.float32)*s[:,None]).reshape(2,64).T
        coarse=np.linalg.norm(reconstruct(64)-w)
        fine=np.linalg.norm(reconstruct(16)-w)
        self.assertLess(fine,coarse)
        np.testing.assert_allclose(reconstruct(16)[16:],w[16:],atol=1e-6)

    def test_standard_onnx_weight_only_and_external_compaction(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);src=p/"decoder.onnx";out=p/"decoder.int8.onnx"
            rng=np.random.default_rng(10)
            weight=rng.normal(size=(128,32)).astype(np.float32)
            other=rng.normal(size=(32,8)).astype(np.float32)
            graph=h.make_graph([
                h.make_node("MatMul",["x","w"],["hidden"],name="/ffn/inner_proj/MatMul"),
                h.make_node("MatMul",["hidden","other"],["y"],name="/final_proj/MatMul")],
                "fixture",[h.make_tensor_value_info("x",T.FLOAT,[1,128])],
                [h.make_tensor_value_info("y",T.FLOAT,[1,8])],
                [nh.from_array(weight,name="w"),nh.from_array(other,name="other")])
            model=h.make_model(graph,opset_imports=[h.make_opsetid("",17)],ir_version=9)
            onnx.save_model(model,str(src),save_as_external_data=True,all_tensors_to_one_file=True,location="source.weights",size_threshold=0)
            self.assertEqual(convert(src,out,["/ffn/inner_proj/MatMul"]),1)
            m=onnx.load(str(out),load_external_data=False)
            self.assertNotIn("w",[t.name for t in m.graph.initializer])
            self.assertEqual(sum(n.op_type=="DequantizeLinear" for n in m.graph.node),1)
            self.assertFalse(any(n.op_type in ("DynamicQuantizeLinear","MatMulInteger") for n in m.graph.node))
            self.assertLess((p/"decoder.int8.onnx.weights").stat().st_size,(p/"source.weights").stat().st_size)
            # Package must not need the source FP32 external file.
            (p/"source.weights").unlink()
            x=rng.normal(size=(1,128)).astype(np.float32)
            actual=ort.InferenceSession(str(out),providers=["CPUExecutionProvider"]).run(None,{"x":x})[0]
            expected=x@weight@other
            self.assertLess(np.linalg.norm(actual-expected)/np.linalg.norm(expected),0.03)

if __name__=="__main__":
    unittest.main()
