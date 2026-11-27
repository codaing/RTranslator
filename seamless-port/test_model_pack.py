import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T
import model_pack

class ModelPackTest(unittest.TestCase):
    def test_shared_external_file_staged_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src=root/"src";dst=root/"dst";src.mkdir()
            nodes=[h.make_node("MatMul",["x","a"],["y"]),h.make_node("MatMul",["y","b"],["encoded"])]
            graph=h.make_graph(nodes,"fixture",[h.make_tensor_value_info("x",T.FLOAT,[1,8])],
                  [h.make_tensor_value_info("encoded",T.FLOAT,[1,8])],
                  [nh.from_array(np.eye(8,dtype=np.float32),name=k) for k in ("a","b")])
            m=h.make_model(graph,opset_imports=[h.make_opsetid("",17)],ir_version=9)
            onnx.save_model(m,str(src/"encoder.int8.onnx"),save_as_external_data=True,
                all_tensors_to_one_file=True,location="shared.weights",size_threshold=0)
            (src/"encoder.int8.validated.json").write_text(json.dumps(
                {"component":"encoder","parity":"PASS","precision":"int8-dynamic"}))
            with patch.object(model_pack,"copy_unique",wraps=model_pack.copy_unique) as copy:
                model_pack.stage("encoder",src,dst)
                self.assertEqual(sum(args[0].name=="shared.weights" for args,_ in copy.call_args_list),1)
            self.assertEqual((dst/"shared.weights").read_bytes(),(src/"shared.weights").read_bytes())
            onnx.checker.check_model(str(dst/"encoder.int8.onnx"))

    def test_different_content_collision_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);a=p/"a";b=p/"b";a.write_bytes(b"a");b.write_bytes(b"b")
            with self.assertRaises(ValueError):model_pack.copy_unique(a,b)

    def test_hardlink_survives_source_removal(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);a=p/"a";b=p/"b";a.write_bytes(b"weights")
            model_pack.copy_unique(a,b);a.unlink()
            self.assertEqual(b.read_bytes(),b"weights")
