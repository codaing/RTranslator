import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from quantize_int8 import quantize


class QuantizationTest(unittest.TestCase):
    def test_real_int8_graph_and_validation_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            weights = np.eye(8, dtype=np.float32)
            graph = helper.make_graph(
                [helper.make_node("MatMul", ["x", "w"], ["encoded"])],
                "tiny", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8])],
                [helper.make_tensor_value_info("encoded", TensorProto.FLOAT, [1, 8])],
                [numpy_helper.from_array(weights, name="w")])
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
            model.ir_version = 9
            onnx.save(model, str(path / "encoder.onnx"))
            x = np.linspace(-1, 1, 8, dtype=np.float32).reshape(1, 8)
            np.savez(path / "encoder.case0.npz", input_x=x, output_encoded=x @ weights)
            (path / "encoder.validated.json").write_text(json.dumps({
                "component": "encoder", "precision": "fp32", "parity": "PASS", "tests": [{}]}))
            report = json.loads(quantize("encoder", path).read_text())
            self.assertEqual(report["parity"], "PASS")
            self.assertGreater(report["quantized_ops"], 0)


if __name__ == "__main__":
    unittest.main()
