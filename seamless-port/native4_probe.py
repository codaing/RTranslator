"""Validate ORT 1.30 CPU execution and UINT4 nibble packing before model export."""
import tempfile
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper as h,numpy_helper as nh,TensorProto as T
from weight_only_int8 import convert

with tempfile.TemporaryDirectory() as temp:
    root=Path(temp)
    rng=np.random.default_rng(1204)
    w=rng.normal(0.2,0.45,(64,16)).astype(np.float32)
    x=rng.normal(size=(2,64)).astype(np.float32)
    src=root/"source.onnx";out=root/"native4.onnx"
    model=h.make_model(h.make_graph(
        [h.make_node("MatMul",["x","w"],["y"],name="native4_probe")],
        "native4-probe",
        [h.make_tensor_value_info("x",T.FLOAT,[None,64])],
        [h.make_tensor_value_info("y",T.FLOAT,[None,16])],
        [nh.from_array(w,name="w")]),
        opset_imports=[h.make_opsetid("",17)],ir_version=9)
    onnx.save_model(model,str(src),save_as_external_data=True,all_tensors_to_one_file=True,
                    location="source.weights",size_threshold=0)
    convert(src,out,["native4_probe"],group=16,native=True,asymmetric=True,bits=4)
    graph=onnx.load(str(out),load_external_data=False).graph
    node=graph.node[0]
    attrs={a.name:onnx.helper.get_attribute_value(a) for a in node.attribute}
    if attrs.get("bits")!=4 or attrs.get("block_size")!=16:
        raise RuntimeError(f"wrong MatMulNBits attributes: {attrs}")
    options=ort.SessionOptions()
    options.add_session_config_entry("session.disable_prepacking","1")
    session=ort.InferenceSession(str(out),sess_options=options,providers=["CPUExecutionProvider"])
    actual=session.run(None,{"x":x})[0]
    expected=x@w
    error=float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),1e-9))
    print(f"ORT={ort.__version__} providers={ort.get_available_providers()} INT4 relative_error={error:.6f}",flush=True)
    if not np.isfinite(actual).all() or actual.shape!=expected.shape or error>0.12:
        raise RuntimeError("native INT4 MatMulNBits CPU probe failed")
