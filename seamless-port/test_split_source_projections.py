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
