"""Generate a tiny deterministic ONNX test graph, not a speech model."""
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, TensorProto as T, numpy_helper
out = Path("seamless/src/androidTest/assets")
out.mkdir(parents=True, exist_ok=True)
graph = h.make_graph([
    h.make_node("Cast", ["units"], ["float_units"], to=T.FLOAT),
    h.make_node("Mul", ["float_units", "scale"], ["scaled"]),
    h.make_node("Unsqueeze", ["scaled", "axes"], ["waveform"]),
], "synthetic_vocoder_interface", [h.make_tensor_value_info("units", T.INT64, [1, "n"])],
   [h.make_tensor_value_info("waveform", T.FLOAT, [1, 1, "n"])],
   [numpy_helper.from_array(np.array(0.01, dtype=np.float32), name="scale"),
    numpy_helper.from_array(np.array([1], dtype=np.int64), name="axes")])
model = h.make_model(graph, opset_imports=[h.make_opsetid("", 17)])
model.ir_version = 9
onnx.checker.check_model(model)
onnx.save(model, out / "test-vocoder.onnx")
