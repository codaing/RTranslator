import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from quantize_int8 import quantize, select_nodes


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

    def test_decoder_selects_only_ffn_not_attention_or_final_projection(self):
        names=["/model/text_decoder/layers.0/ffn/inner_proj/MatMul",
               "/model/text_decoder/layers.0/ffn/output_proj/MatMul",
               "/model/text_decoder/layers.0/self_attn/output_proj/MatMul",
               "/model/text_decoder/layers.0/encoder_decoder_attn/p_choose/MatMul",
               "/model/final_proj/MatMul"]
        graph=helper.make_graph([helper.make_node("MatMul",["x","w"],[f"y{i}"],name=n)
            for i,n in enumerate(names)],"select",[],[],[numpy_helper.from_array(np.eye(8,dtype=np.float32),name="w")])
        self.assertEqual([e["name"] for e in select_nodes(graph,"decoder") if e["selected"]],names[:2])
        self.assertEqual(len([e for e in select_nodes(graph,"encoder") if e["selected"]]),5)


if __name__ == "__main__":
    unittest.main()
