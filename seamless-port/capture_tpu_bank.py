"""Capture a bank of actual SeamlessStreaming source projection matrices.

Group only byte-identical real activations of independent MatMul nodes.
Concatenate columns, preserving individual outputs and original weight values.
A bank is a representative subgraph, not the complete source-projection stage.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
from export_vulkan_projection_fixture import (_candidate_nodes,_constant_array,
    _capture_value_info,_nrms)

def capture(directory,features,output):
    d=Path(directory).resolve();out=Path(output);out.mkdir(parents=True,exist_ok=True)
    opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1
    f=np.fromfile(features,dtype="<f4").reshape(1,-1,80)
    enc=ort.InferenceSession(str(d/"encoder.int8.onnx"),opts,providers=["CPUExecutionProvider"])
    encoded=enc.run(["encoded"],{enc.get_inputs()[0].name:f})[0];del enc
    model=onnx.load(str(d/"decoder.source-projections.onnx"),load_external_data=True)
    candidates,initializers,producers=_candidate_nodes(model)
    candidates=[c for c in candidates if c[5:]==(1024,1024)]
    if not candidates:raise ValueError("no real 1024x1024 MatMul")
    names=list(dict.fromkeys(n for c in candidates for n in (c[2],c[1].output[0])))
    probe=d/"decoder.tpu-capture.onnx"
    onnx.save_model(_capture_value_info(model,names),str(probe))
    try:
        s=ort.InferenceSession(str(probe),opts,providers=["CPUExecutionProvider"])
        values=dict(zip(names,s.run(names,{s.get_inputs()[0].name:encoded})));del s
    finally:probe.unlink(missing_ok=True)
    groups=collections.defaultdict(list)
    for c in candidates:
        a=np.ascontiguousarray(values[c[2]].reshape(-1,1024),dtype=np.float32)
        groups[(a.shape,hashlib.sha256(a.tobytes()).hexdigest())].append(c)
    group=max(groups.values(),key=len)
    if len(group)<2:raise ValueError("no shared real activation for bank")
    group=group[:8]
    x=np.ascontiguousarray(values[group[0][2]].reshape(-1,1024),dtype=np.float32)
    weights=[];original=[];nodes=[]
    for c in group:
        a=np.ascontiguousarray(values[c[2]].reshape(-1,1024),dtype=np.float32)
        if not np.array_equal(a,x):raise ValueError("activation equality failed")
        w=_constant_array(c[3],initializers,producers,d)
        if w.shape!=(1024,1024) or w.dtype!=np.float16:raise ValueError("unexpected weights")
        y=values[c[1].output[0]].reshape(x.shape[0],1024).astype(np.float32)
        if not all(np.isfinite(v).all() for v in (x,w,y)):raise ValueError("nonfinite data")
        weights.append(w);original.append(y);nodes.append(c[1].name)
    for label,count in (("single",1),("bank",len(group))):
        w=np.ascontiguousarray(np.concatenate(weights[:count],axis=1),dtype=np.float32)
        y=np.ascontiguousarray(x@w,dtype=np.float32)
        original_y=np.concatenate(original[:count],axis=1)
        meta={"name":label,"M":x.shape[0],"K":x.shape[1],"N":w.shape[1],"actual_projections":count,
              "nodes":nodes[:count],"source_fp16_vs_fp32_nrms":_nrms(original_y,y),
              "input_sha256":hashlib.sha256(x.tobytes()).hexdigest(),
              "weight_sha256":hashlib.sha256(w.tobytes()).hexdigest(),
              "source_model_sha256":hashlib.sha256((d/"decoder.source-projections.onnx").read_bytes()).hexdigest(),
              "calibration":"same real Mandarin fixture; feasibility only",
              "license":"Meta SeamlessStreaming CC BY-NC 4.0"}
        np.savez(out/(label+".npz"),x=x,w=w,y=y)
        (out/(label+".json")).write_text(json.dumps(meta,indent=2)+"\n")
        print(json.dumps(meta),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("directory");p.add_argument("features");p.add_argument("output")
    a=p.parse_args();capture(a.directory,a.features,a.output)
