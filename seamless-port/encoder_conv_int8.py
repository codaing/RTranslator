"""Experimental calibrated INT8 encoder convolution, preserving full context.

Only existing FP32 Conv nodes are selected. The already validated dynamic
INT8 MatMul graph and all attention, normalization and decoder operations
remain unchanged. Quantized candidates are not released as model packs.
"""
import argparse
import collections
import json
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto as T
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod,
    QuantFormat, QuantType, quantize_static)

GRAPH="encoder.int8.onnx"
PROFILES={
    "qdq-all": (QuantFormat.QDQ,QuantType.QInt8,"all"),
    "qoperator-all": (QuantFormat.QOperator,QuantType.QUInt8,"all"),
    "qoperator-pointwise": (QuantFormat.QOperator,QuantType.QUInt8,"pointwise"),
}

def selection(model,scope):
    g=model.graph
    weights={t.name:t for t in g.initializer}
    nodes=[]
    for n in g.node:
        if n.op_type!="Conv" or n.domain not in ("","ai.onnx"):continue
        if len(n.input)<2 or n.input[1] not in weights:
            raise ValueError("nonconstant Conv weights: "+n.name)
        w=weights[n.input[1]]
        if w.data_type!=T.FLOAT or len(w.dims)<3:
            raise ValueError("unsupported Conv weight: "+n.name)
        group=next((a.i for a in n.attribute if a.name=="group"),1)
        kernel=list(w.dims[2:])
        kind="pointwise" if all(k==1 for k in kernel) and group==1 else "spatial-or-grouped"
        if scope=="all" or kind==scope:
            nodes.append({"name":n.name,"shape":list(w.dims),"group":group,"kind":kind})
    if not nodes or any(not n["name"] for n in nodes) or len({n["name"] for n in nodes})!=len(nodes):
        raise ValueError("invalid convolution selection")
    return nodes

class PrefixReader(CalibrationDataReader):
    def __init__(self,features,lengths):
        self.features=np.ascontiguousarray(features,dtype=np.float32)
        if self.features.ndim!=2 or self.features.shape[1]!=80 or not np.isfinite(self.features).all():
            raise ValueError("invalid filterbank")
        self.lengths=tuple(sorted(set(lengths)))
        if not self.lengths or any(n<2 or n%2 or n>len(features) for n in self.lengths):
            raise ValueError("invalid calibration prefix")
        self.rewind()
    def get_next(self):
        n=next(self.iterator,None)
        if n is None:return None
        return {"fbank":self.features[None,:n,:].copy()}
    def rewind(self):self.iterator=iter(self.lengths)

def calibrate(source,target,features,profile,expected_total=74):
    source,target=Path(source),Path(target)
    format,activation,scope=PROFILES[profile]
    model=onnx.load(str(source),load_external_data=False)
    total=selection(model,"all")
    if expected_total is not None and len(total)!=expected_total:
        raise ValueError("unexpected encoder Conv inventory: "+str(len(total)))
    selected=selection(model,scope)
    lengths=list(range(192,len(features)//2*2+1,128))
    end=len(features)//2*2
    if end<192:raise ValueError("calibration recording too short")
    lengths.append(end)
    reader=PrefixReader(features,lengths)
    original_counts=collections.Counter((n.domain,n.op_type) for n in model.graph.node)
    del model
    quantize_static(str(source),str(target),reader,quant_format=format,
        activation_type=activation,weight_type=QuantType.QInt8,per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,nodes_to_quantize=[n["name"] for n in selected],
        op_types_to_quantize=["Conv"],use_external_data_format=True,
        extra_options={"WeightSymmetric":True,"ActivationSymmetric":activation==QuantType.QInt8})
    onnx.checker.check_model(str(target))
    candidate=onnx.load(str(target),load_external_data=False)
    g=candidate.graph
    counts=collections.Counter((n.domain,n.op_type) for n in g.node)
    for op in ("MatMulInteger","DynamicQuantizeLinear"):
        if sum(v for (d,k),v in original_counts.items() if k==op)!=sum(v for (d,k),v in counts.items() if k==op):
            raise ValueError("existing dynamic INT8 matrix operators changed: "+op)
    if format==QuantFormat.QOperator:
        actual=sum(v for (d,k),v in counts.items() if k=="QLinearConv")
        if actual!=len(selected):raise ValueError(f"expected {len(selected)} QLinearConv, got {actual}")
    else:
        by_name={n.name:n for n in g.node}
        producers={o:n for n in g.node for o in n.output}
        for row in selected:
            node=by_name.get(row["name"])
            if node is None or node.op_type!="Conv" or any(producers.get(x) is None or producers[x].op_type!="DequantizeLinear" for x in node.input[:2]):
                raise ValueError("missing QDQ convolution inputs: "+row["name"])
    return {"profile":profile,"selected":len(selected),"conv_inventory":len(total),
            "calibration_prefix_frames":reader.lengths,"calibration_scope":"same speech fixture; not held-out accuracy",
            "selected_nodes":selected,"candidate_ops":{d+"/"+k:v for (d,k),v in counts.items()}}

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source");p.add_argument("target");p.add_argument("features")
    p.add_argument("--profile",choices=PROFILES,required=True)
    a=p.parse_args()
    print(json.dumps(calibrate(a.source,a.target,np.fromfile(a.features,dtype="<f4").reshape(-1,80),a.profile),indent=2))
