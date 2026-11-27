"""Regression for the exact frozen shape pattern observed in CI."""
import unittest
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper as h, numpy_helper as nh, TensorProto as T
from repair_encoder_graph import rewrite


class TestRepair(unittest.TestCase):
    def test_frozen_positions_and_broadcast(self):
        p = "/speech_encoder/inner/layers.0/self_attn/sdpa"
        def n(op, ins, out, suffix, **kw):
            return h.make_node(op, ins, [out], name=p+suffix, **kw)
        def const(out, value, suffix):
            return n("Constant",[],out,suffix,value=nh.from_array(np.array(value,dtype=np.int64)))
        nodes = [
            const("q_ones",np.ones(96),"/Constant"),
            const("k_ones",np.ones(96),"/Constant_3"),
            const("axis",0,"/Axis"),
            n("CumSum",["q_ones","axis"],"qpos","/CumSum"),
            n("CumSum",["k_ones","axis"],"kpos","/CumSum_1"),
            const("qaxes",[1],"/QAxes"),const("kaxes",[0],"/KAxes"),
            n("Unsqueeze",["qpos","qaxes"],"qr","/QRow"),
            n("Unsqueeze",["kpos","kaxes"],"kr","/KRow"),
            n("Sub",["kr","qr"],"distance","/Distance"),
            const("low",-2,"/Low"),const("high",2,"/High"),
            n("Clip",["distance","low","high"],"clipped","/Clip"),
            const("offset",2,"/Offset"),
            n("Add",["clipped","offset"],"idx","/OffsetAdd"),
            const("axes",[0,1],"/Axes"),
            n("Unsqueeze",["idx","axes"],"idx4","/Unsqueeze"),
            const("zeros",np.zeros((1,2,96,1)),"/Constant_13"),
            n("Add",["idx4","zeros"],"indices","/Add_2"),
            n("MatMul",["q","table"],"rel","/MatMul"),
            n("MatMul",["q","kt"],"content","/MatMul_1"),
            n("GatherElements",["rel","indices"],"relative","/GatherElements",axis=-1),
            n("Add",["content","relative"],"output","/Add_3"),
        ]
        graph = h.make_graph(nodes,"regression",
            [h.make_tensor_value_info("q",T.FLOAT,[1,2,"time",4]),
             h.make_tensor_value_info("kt",T.FLOAT,[1,2,4,"time"])],
            [h.make_tensor_value_info("output",T.FLOAT,[1,2,"time","time"])],
            [nh.from_array(np.arange(20,dtype=np.float32).reshape(4,5),name="table")])
        model=h.make_model(graph,opset_imports=[h.make_opsetid("",17)],ir_version=9)
        self.assertEqual(rewrite(model.graph,expected_layers=1),3)
        onnx.checker.check_model(model)
        sess=ort.InferenceSession(model.SerializeToString(),providers=["CPUExecutionProvider"])
        rng=np.random.default_rng(0)
        table=np.arange(20,dtype=np.float32).reshape(4,5)
        for length in (96,128,97,1):
            q=rng.normal(size=(1,2,length,4)).astype(np.float32)
            kt=rng.normal(size=(1,2,4,length)).astype(np.float32)
            idx=np.clip(np.arange(length)[None,:]-np.arange(length)[:,None],-2,2)+2
            expected=q@kt+np.take_along_axis(q@table,np.broadcast_to(idx,(1,2,length,length)),axis=-1)
            np.testing.assert_allclose(sess.run(None,{"q":q,"kt":kt})[0],expected,rtol=1e-5,atol=1e-5)


if __name__ == "__main__":
    unittest.main()
