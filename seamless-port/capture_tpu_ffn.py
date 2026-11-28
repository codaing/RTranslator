"""Capture a genuine decoder FFN core from actual speech inference.

Extract the dependency path between expansion input and contraction output
(including a constant bias add after contraction if present). Reject additional
dynamic inputs or unexpected operators. Keep original ONNX weights external.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper as nh
import onnxruntime as ort
from export_vulkan_projection_fixture import _candidate_nodes, _constant_array, _capture_value_info

OPS={"MatMul","Add","Relu","Gelu","Erf","Mul","Div","Sub","Pow","Tanh","Sigmoid","Identity","Cast","Neg","Sqrt"}

def plan(model,directory):
    candidates,init,producers=_candidate_nodes(model)
    expand=[c for c in candidates if c[5:]==(1024,8192)]
    contract=[c for c in candidates if c[5:]==(8192,1024)]
    if len(expand)!=24 or len(contract)!=24:
        raise ValueError("unexpected FFN inventory")
    constants={};steps=[];visited=set()
    def constant(name):
        if name in constants:return constants[name]
        v=_constant_array(name,init,producers,directory)
        if v is None:
            node=producers.get(name)
            if node is not None and node.op_type=="Constant":
                a=next((a for a in node.attribute if a.name=="value"),None)
                if a is not None:v=nh.to_array(a.t)
        if v is not None:
            v=np.asarray(v)
            if not np.issubdtype(v.dtype,np.floating) or not np.isfinite(v).all():
                raise ValueError("unexpected FFN constant "+name)
            constants[name]=np.ascontiguousarray(v,dtype=np.float32) if v.ndim else v.astype(np.float32)
        return v
    def trace(name,x):
        if name==x or name in visited:return
        if constant(name) is not None:visited.add(name);return
        node=producers.get(name)
        if node is None or node.op_type not in OPS or node.domain not in ("","ai.onnx","com.microsoft") or len(node.output)!=1:
            raise ValueError("unexpected FFN dependency "+name)
        for a in node.input:
            if not a:raise ValueError("empty input")
            trace(a,x)
        attrs={a.name:h.get_attribute_value(a) for a in node.attribute}
        # Casts inside the FP16 original are represented by FP32 in the
        # conversion. Report their rounding difference against the real output.
        if node.op_type=="Cast" and attrs.get("to") not in (T.FLOAT,T.FLOAT16):
            raise ValueError("non-floating FFN cast")
        if node.op_type=="Gelu":
            approx=attrs.get("approximate",b"none")
            if approx not in (b"none",b"tanh"):raise ValueError("unknown GELU approximation")
            attrs={"approximate":approx.decode()}
        elif node.op_type=="Cast":attrs={"to":attrs["to"]}
        elif attrs:raise ValueError("unhandled attributes "+node.op_type)
        steps.append({"op":node.op_type,"name":node.name,"inputs":list(node.input),"output":name,"attrs":attrs})
        visited.add(name)
    failures=[]
    for e in expand:
        x=e[2];first=e[1].output[0]
        for c in contract:
            constants.clear();steps.clear();visited.clear()
            final=c[1].output[0]
            try:
                trace(final,x)
                mat=[s for s in steps if s["op"]=="MatMul"]
                if [s["output"] for s in mat]!=[first,final]:
                    continue
                if not any(s["op"] in ("Relu","Gelu","Erf","Sigmoid","Tanh") for s in steps):
                    raise ValueError("no activation in FFN")
                # Include a direct constant bias if unambiguously present.
                bias_nodes=[n for n in model.graph.node if n.op_type=="Add" and final in n.input and len(n.input)==2]
                if len(bias_nodes)==1:
                    b=bias_nodes[0];other=next(i for i in b.input if i!=final)
                    if constant(other) is not None:
                        trace(b.output[0],x);final=b.output[0]
                for s in mat:
                    if s["inputs"][1] not in constants:raise ValueError("dynamic FFN weights")
                return x,final,steps.copy(),constants.copy()
            except ValueError as exc:
                failures.append(str(exc))
    raise ValueError("no verified FFN core: "+str(failures[:5]))

def capture(directory,features,output):
    from speech_smoke import run
    d=Path(directory).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True)
    graph=d/"decoder.cached-step.onnx";model=onnx.load(str(graph),load_external_data=False)
    x,y,steps,constants=plan(model,d)
    probe=d/"decoder.ffn-tpu-capture.onnx"
    onnx.save_model(_capture_value_info(model,[x,y]),str(probe))
    del model
    keys={name:f"constant_{i}" for i,name in enumerate(constants)}
    np.savez(out/"weights.npz",**{keys[n]:v for n,v in constants.items()})
    metadata={"input":x,"output":y,"steps":steps,"constants":keys,
        "source_graph_sha256":hashlib.sha256(graph.read_bytes()).hexdigest(),
        "source":"real cached-step decoder; original FP16 output","cases":[]}
    (out/"plan.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps({"steps":steps,"input":x,"output":y}),flush=True)
    real=ort.InferenceSession
    class Session:
        def __init__(self,path,*args,**kwargs):
            self.capture=Path(path).name==graph.name
            self.inner=real(str(probe) if self.capture else path,*args,**kwargs)
            self.calls=0
        def __getattr__(self,name):return getattr(self.inner,name)
        def run(self,names,feeds,*args,**kwargs):
            if not self.capture:return self.inner.run(names,feeds,*args,**kwargs)
            self.calls+=1
            if names is None:raise ValueError("unexpected implicit decoder outputs")
            extra=[n for n in (x,y) if n not in names]
            results=self.inner.run(list(names)+extra,feeds,*args,**kwargs)
            values=dict(zip(list(names)+extra,results))
            a,b=values[x],values[y]
            if a.shape[-1]!=1024 or a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError("unexpected FFN activation")
            a=a.reshape(-1,1024).astype(np.float32);b=b.reshape(-1,1024).astype(np.float32)
            case=f"case-{self.calls:04d}.npz"
            np.savez(out/case,x=a,y=b)
            metadata["cases"].append({"file":case,"rows":len(a)})
            return results[:len(names)]
    ort.InferenceSession=Session
    try:run(d,features,str(out/"speech.wav"),source_projections=True)
    finally:
        ort.InferenceSession=real
        probe.unlink(missing_ok=True)
    if len(metadata["cases"])<5:raise ValueError("not enough real decoder calls")
    (out/"plan.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps({"cases":len(metadata["cases"]),"row_counts":sorted(set(c["rows"] for c in metadata["cases"]))}),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("directory");p.add_argument("features");p.add_argument("output")
    a=p.parse_args();capture(a.directory,a.features,a.output)
