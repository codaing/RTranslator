"""Static real-model inventory for GPU planning, not provider placement evidence."""
import collections
import json
import math
from pathlib import Path
import sys
import onnx
from onnx import TensorProto

NAMES=("encoder.int8.onnx","decoder.cached-step.onnx","decoder.source-projections.onnx","units.onnx","vocoder.onnx")
def audit(path):
    g=onnx.load(str(path),load_external_data=False).graph
    weights={t.name:t for t in g.initializer}
    ops=collections.Counter((n.domain or "ai.onnx")+":"+n.op_type for n in g.node)
    types=collections.Counter()
    largest=[]
    for t in g.initializer:
        typ=TensorProto.DataType.Name(t.data_type)
        elems=math.prod(t.dims)
        types[typ]+=elems
        if len(t.dims)>=2:
            largest.append({"name":t.name,"dtype":typ,"dims":list(t.dims),"elements":elems})
    linear=collections.Counter()
    for n in g.node:
        if n.op_type not in ("MatMul","Gemm","MatMulInteger","QLinearMatMul","MatMulNBits","Conv","QLinearConv","ConvInteger"):continue
        key=(n.domain or "ai.onnx")+":"+n.op_type
        linear[key]+=1
    return {"graph":path.name,"nodes":len(g.node),"operators":dict(ops.most_common()),
            "linear_ops":dict(linear),"initializer_elements_by_dtype":dict(types),
            "largest_initializers":sorted(largest,key=lambda x:x["elements"],reverse=True)[:15],
            "scope":"static graph inventory; no claims about kernel support, GPU placement or throughput"}

def main(directory,output):
    rows=[audit(Path(directory)/n) for n in NAMES]
    Path(output).write_text(json.dumps(rows,indent=2)+"\n")
    for row in rows:
        print(json.dumps(row),flush=True)
if __name__=="__main__":main(*sys.argv[1:])
