import unittest
import numpy as np
from onnx import helper as h, TensorProto as T
from encoder_cache_audit import attention_plan,expose,drift,future_mass

class CacheAuditTest(unittest.TestCase):
    def test_drift_prefix_and_later_stable_rows(self):
        a=np.arange(24,dtype=np.float32).reshape(1,2,3,4)
        b=np.concatenate((a,np.ones((1,2,1,4),np.float32)),axis=2)
        b[:,:,1,:]+=1
        r=drift(a,b,2)
        self.assertEqual(r["exact_prefix"],1)
        self.assertEqual(r["exact_rows"],2)
        self.assertEqual(r["tolerance_prefix"],1)
        self.assertGreater(r["nrms"],0)
        self.assertEqual(drift(a.transpose(0,1,3,2),b.transpose(0,1,3,2),3),r)
    def test_future_mass(self):
        p=np.zeros((1,2,4,4),np.float32)
        p[:,:,:,0]=1
        p[:,:,1,0]=.75
        p[:,:,1,3]=.25
        r=future_mass(p,3)
        self.assertEqual(r["first_affected_query"],1)
        self.assertEqual(r["queries_with_new_key_mass_gt_1e-7"],1)
        self.assertEqual(r["max_mass"],.25)
        self.assertIsNone(future_mass(p,1)["first_affected_query"])
    def test_exact_attention_dataflow(self):
        pre="/model/speech_encoder/inner/layers.0/self_attn/sdpa/"
        ns=[h.make_node("MatMul",["q","kt"],["scores"],name=pre+"MatMul_1"),
            h.make_node("Softmax",["scores"],["p"],name=pre+"Softmax"),
            h.make_node("Cast",["p"],["pc"],name=pre+"Cast",to=T.FLOAT),
            h.make_node("MatMul",["pc","v"],["c"],name=pre+"MatMul_2")]
        g=h.make_graph(ns,"test",[h.make_tensor_value_info(k,T.FLOAT,[None]*4) for k in ("q","kt","v")],
                       [h.make_tensor_value_info("c",T.FLOAT,[None]*4)])
        m=h.make_model(g)
        plan=attention_plan(m,1)
        self.assertEqual(plan,[{"layer":0,"q":"q","kt":"kt","v":"v","prob":"p","context":"c"}])
        self.assertEqual({v.name for v in expose(m,plan).graph.output},{"c","q","kt","v","p"})
        self.assertEqual(len(m.graph.output),1)
        with self.assertRaises(ValueError):attention_plan(m,2)
if __name__=="__main__":unittest.main()
