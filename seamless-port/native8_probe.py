"""Verify both native INT8 packing layouts before a multi-gigabyte conversion."""
import tempfile
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper as h,numpy_helper as nh,TensorProto as T
from weight_only_int8 import convert

with tempfile.TemporaryDirectory() as temp:
    root=Path(temp)
    rng=np.random.default_rng(12)
    w=rng.normal(0.25,0.5,(64,16)).astype(np.float32)
    x=rng.normal(size=(2,64)).astype(np.float32)
    src=root/"source.onnx"
    model=h.make_model(h.make_graph([h.make_node("MatMul",["x","w"],["y"],name="test")],
       "probe",[h.make_tensor_value_info("x",T.FLOAT,[None,64])],
       [h.make_tensor_value_info("y",T.FLOAT,[None,16])],[nh.from_array(w,name="w")]),
       opset_imports=[h.make_opsetid("",17)],ir_version=9)
    onnx.save_model(model,str(src),save_as_external_data=True,location="original.weights",size_threshold=0)
    for asym in (False,True):
        out=root/f"native-{asym}.onnx"
        convert(src,out,["test"],group=16,native=True,asymmetric=asym)
        opts=ort.SessionOptions()
        opts.add_session_config_entry("session.disable_prepacking","1")
        session=ort.InferenceSession(str(out),opts,providers=["CPUExecutionProvider"])
        actual=session.run(None,{"x":x})[0]
        expected=x@w
        error=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
        print(f"native8 asymmetric={asym} relative error={error:.6f}",flush=True)
        if error>0.02:raise RuntimeError("native packed layout failed")
