import unittest
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T
from split_source_projections import partition

class SplitSourceTest(unittest.TestCase):
    def test_hoist_only_source_matmul(self):
        def vi(n):return h.make_tensor_value_info(n,T.FLOAT,[1,2])
        g=h.make_graph([
            h.make_node("MatMul",["encoded","w"],["projected"],name="source_proj"),
            h.make_node("MatMul",["target","w"],["target_proj"],name="target_proj"),
            h.make_node("Add",["projected","target_proj"],["y"],name="mix")],
            "test",[vi("encoded"),vi("target")],[vi("y")],
            [nh.from_array(np.eye(2,dtype=np.float32),name="w")])
        m=h.make_model(g,opset_imports=[h.make_opsetid("",17)],ir_version=9)
        source,main,report=partition(m)
        self.assertEqual(report["source_matmuls"],1)
        self.assertEqual(report["boundary"],["projected"])
        self.assertEqual([n.name for n in source.graph.node],["source_proj"])
        self.assertEqual([n.name for n in main.graph.node],["target_proj","mix"])
        onnx.checker.check_model(source);onnx.checker.check_model(main)
        from onnx.reference import ReferenceEvaluator
        e=np.array([[1.,2.]],np.float32);t=np.array([[3.,4.]],np.float32)
        expected=ReferenceEvaluator(m).run(None,{"encoded":e,"target":t})[0]
        p=ReferenceEvaluator(source).run(None,{"encoded":e})[0]
        actual=ReferenceEvaluator(main).run(None,{"projected":p,"target":t})[0]
        np.testing.assert_array_equal(expected,actual)

    def test_external_reshape_shape_metadata(self):
        import tempfile
        from pathlib import Path
        def vi(n):return h.make_tensor_value_info(n,T.FLOAT,[1,2])
        g=h.make_graph([
            h.make_node("MatMul",["encoded","w"],["projected"],name="source_proj"),
            h.make_node("Reshape",["projected","shape"],["reshaped"]),
            h.make_node("Add",["reshaped","target"],["y"])],
            "external_shape",[vi("encoded"),vi("target")],[vi("y")],
            [nh.from_array(np.eye(2,dtype=np.float32),name="w"),
             nh.from_array(np.array([1,2],np.int64),name="shape")])
        m=h.make_model(g,opset_imports=[h.make_opsetid("",17)],ir_version=9)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"original.onnx"
            onnx.save_model(m,str(p),save_as_external_data=True,all_tensors_to_one_file=True,
                location="weights.bin",size_threshold=0)
            original=onnx.load(str(p),load_external_data=False)
            source,main,report=partition(original,tmp)
            self.assertEqual(report["boundary"],["reshaped"])
            self.assertTrue(all(t.data_location==T.EXTERNAL for t in original.graph.initializer))
            for name,model in (("source",source),("main",main)):
                onnx.save_model(model,str(Path(tmp)/(name+".onnx")))
                onnx.checker.check_model(str(Path(tmp)/(name+".onnx")))
