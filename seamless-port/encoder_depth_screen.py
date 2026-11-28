"""Training-free depth compression screen for the real Seamless encoder.

This deliberately changes the model; it is not exact inference optimization.
Retain original frontend, adapter, first and last blocks and all source frames.
Never publish or deploy these untrained student candidates.
"""
import argparse
import collections
import json
from pathlib import Path
import re
import numpy as np
import onnx
from onnx import TensorProto as T
from split_source_projections import deps

GRAPH="encoder.int8.onnx"
LAYER=re.compile(r"/speech_encoder/inner/layers\.(\d+)(?:/|$)")
PROFILES={"keep18":tuple(i for i in range(24) if i not in (3,6,9,12,15,18)),
          "keep12":tuple([0]+list(range(2,22,2))+[23]),
          "keep6":(0,5,10,15,20,23)}

def layer_id(node):
    m=LAYER.search(node.name)
    return int(m.group(1)) if m else None

def boundaries(model,expected_layers=24):
    g=model.graph
    consumers=collections.defaultdict(list)
    groups=collections.defaultdict(list)
    for node in g.node:
        idx=layer_id(node)
        if idx is not None:groups[idx].append(node)
        for name in deps(node):consumers[name].append(node)
    if sorted(groups)!=list(range(expected_layers)):
        raise ValueError("unexpected encoder layer inventory: "+str(sorted(groups)))
    outputs={}
    # Verify each consecutive pair of residual blocks is connected through
    # exactly one final normalization activation. Do not pick a node by index.
    for idx in range(expected_layers-1):
        candidates=[]
        for node in groups[idx]:
            if node.op_type not in ("LayerNormalization","Add"):continue
            for name in node.output[:1]:
                if any(layer_id(c)==idx+1 for c in consumers[name]):
                    candidates.append((node,name))
        norm=[(n,x) for n,x in candidates if n.op_type=="LayerNormalization"]
        selected=norm if norm else candidates
        if len(selected)!=1:
            raise ValueError(f"layer {idx}: ambiguous residual interface: "+str([(n.name,x) for n,x in candidates]))
        outputs[idx]=selected[0][1]
    return outputs

def rewrite(model,keep,expected_layers=24):
    keep=tuple(keep)
    if sorted(set(keep))!=list(keep) or not keep or keep[0]!=0 or keep[-1]!=expected_layers-1:
        raise ValueError("retain first and last encoder blocks and ordered unique IDs")
    if any(i<0 or i>=expected_layers for i in keep):raise ValueError("invalid kept block")
    g=model.graph
    boundary=boundaries(model,expected_layers)
    drop=set(range(expected_layers))-set(keep)
    aliases={}
    for i in sorted(drop):
        previous=max(k for k in keep if k<i)
        aliases[boundary[i]]=boundary[previous]
    # Substitute the residual output only, then reverse-reachability prune.
    # If hidden cross-layer dependencies retain any dropped block, fail closed.
    for node in g.node:
        for j,x in enumerate(node.input):
            if x in aliases:node.input[j]=aliases[x]
        if (deps(node)-set(node.input))&set(aliases):
            raise ValueError("nested subgraph captures removed activation")
    if {o.name for o in g.output}&set(aliases):raise ValueError("cannot bypass graph output")
    needed={o.name for o in g.output}
    live=[]
    for node in reversed(g.node):
        if needed.intersection(node.output):
            live.append(node);needed.update(deps(node))
    live.reverse()
    remaining=sorted({layer_id(n) for n in live if layer_id(n) is not None})
    if remaining!=list(keep):
        raise ValueError("hidden dependencies prevent exact block removal: "+str(remaining))
    original_nodes=len(g.node)
    g.ClearField("node");g.node.extend(live)
    retained=[t for t in g.initializer if t.name in needed]
    original_bytes=sum(int(np.prod(t.dims))*{T.FLOAT:4,T.FLOAT16:2,T.INT8:1,T.UINT8:1,T.INT64:8,T.INT32:4}.get(t.data_type,0) for t in g.initializer)
    retained_bytes=sum(int(np.prod(t.dims))*{T.FLOAT:4,T.FLOAT16:2,T.INT8:1,T.UINT8:1,T.INT64:8,T.INT32:4}.get(t.data_type,0) for t in retained)
    g.ClearField("initializer");g.initializer.extend(retained)
    valid=needed|{o for n in live for o in n.output}
    for vi in list(g.value_info):
        if vi.name not in valid:g.value_info.remove(vi)
    return {"kept":list(keep),"removed":sorted(drop),"original_nodes":original_nodes,"retained_nodes":len(live),
            "original_logical_initializer_bytes":original_bytes,"retained_logical_initializer_bytes":retained_bytes,
            "bypasses":aliases,"note":"External weight file not compacted; logical bytes are not APK size."}

def create(directory,profile):
    d=Path(directory)
    model=onnx.load(str(d/GRAPH),load_external_data=False)
    report=rewrite(model,PROFILES[profile])
    target=d/("encoder.depth-"+profile+".onnx")
    onnx.save_model(model,str(target));onnx.checker.check_model(str(target))
    return target,report

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("directory");p.add_argument("profile",choices=PROFILES)
    a=p.parse_args();target,report=create(a.directory,a.profile);print(json.dumps(report,indent=2))
