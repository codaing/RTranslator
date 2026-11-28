"""Experimental dynamic INT8 FFN for the actual projected decoder.

Preserve all token IDs, attention, monotonic policy, caches, feature outputs and
original graph IO. Only constant-weight FFN MatMuls change precision. Keep the
existing external weights; write a separate candidate INT8 file. Never promote
from matrix parity alone.
"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper as nh
from split_source_projections import deps

SHAPES=((1024,8192),(8192,1024))
GRAPH="decoder.cached-step.onnx"

def quantize_columns(w):
    w=np.asarray(w,dtype=np.float32)
    if w.ndim!=2 or not np.isfinite(w).all():raise ValueError("invalid FFN weight")
    s=np.max(np.abs(w),axis=0)/np.float32(127)
    s=np.where(s>0,s,np.float32(1)).astype(np.float32)
    q=np.clip(np.rint(w/s),-127,127).astype(np.int8)
    return np.ascontiguousarray(q),s

def rewrite(model,source,weight_file,selected_shapes=SHAPES,expected_per_shape=24):
    g=model.graph
    init={t.name:t for t in g.initializer}
    producers={o:n for n in g.node for o in n.output if o}
    # Resolve only constant rank-two weights with verified Cast/Transpose
    # semantics, using metadata first so large unrelated tensors never load.
    def trace(name,seen=()):
        if name in seen:return None
        if name in init:
            t=init[name]
            if len(t.dims)!=2 or t.data_type not in (T.FLOAT,T.FLOAT16):return None
            return t,tuple(t.dims),t.data_type,()
        n=producers.get(name)
        if n is None or n.domain not in ("","ai.onnx") or n.op_type not in ("Transpose","Cast","Identity") or len(n.input)!=1:return None
        result=trace(n.input[0],seen+(name,))
        if result is None:return None
        t,shape,dtype,ops=result
        if n.op_type=="Transpose":
            perm=next((list(a.ints) for a in n.attribute if a.name=="perm"),[1,0])
            if perm not in ([1,0],[0,1]):return None
            shape=tuple(shape[i] for i in perm);ops+=(("transpose",perm),)
        elif n.op_type=="Cast":
            dtype=next((a.i for a in n.attribute if a.name=="to"),0)
            if dtype not in (T.FLOAT,T.FLOAT16):return None
            ops+=(("cast",dtype),)
        return t,shape,dtype,ops
    chosen={}
    counts={shape:0 for shape in selected_shapes}
    for i,n in enumerate(g.node):
        if n.domain not in ("","ai.onnx") or n.op_type!="MatMul" or len(n.input)!=2:continue
        info=trace(n.input[1])
        if info is None or info[1] not in counts:continue
        # These shapes uniquely identify the 24 layers' two FFN projections
        # in the validated 1024/8192 decoder. Require exact counts, not a
        # broad quantization of attention or the full-vocabulary head.
        counts[info[1]]+=1
        chosen[i]=info
    if not chosen or any(v!=expected_per_shape for v in counts.values()):
        raise ValueError("unexpected FFN matrix inventory: "+str(counts))
    names={x.name for x in g.initializer}|{x.name for x in g.input}|set(producers)
    if any(n.startswith("ffn_dyn8_") for n in names) or any(n.name.startswith("ffn_dyn8_") for n in g.node):
        raise ValueError("candidate name collision")
    new_nodes=[];new_weights=[];stats=[]
    weight_file=Path(weight_file)
    with weight_file.open("wb") as out:
        for i,n in enumerate(g.node):
            if i not in chosen:
                new_nodes.append(n);continue
            t,shape,dtype,ops=chosen[i]
            w=nh.to_array(copy.deepcopy(t),base_dir=str(source))
            for op,arg in ops:
                if op=="transpose":w=w.transpose(arg)
                else:w=w.astype(np.float16 if arg==T.FLOAT16 else np.float32)
            if w.shape!=shape:raise ValueError("weight layout mismatch")
            q,s=quantize_columns(w)
            prefix=f"ffn_dyn8_{i}_"
            offset=out.tell();out.write(q.tobytes())
            qt=T(name=prefix+"weight",data_type=T.INT8,dims=list(shape),data_location=T.EXTERNAL)
            for k,v in (("location",weight_file.name),("offset",str(offset)),("length",str(q.nbytes))):
                e=qt.external_data.add();e.key=k;e.value=v
            new_weights.extend([qt,nh.from_array(s,name=prefix+"wscale"),
                nh.from_array(np.array(0,dtype=np.int8),name=prefix+"wzero")])
            def node(op,inputs,outputs,**kw):
                return h.make_node(op,inputs,outputs,name=prefix+op,**kw)
            x=n.input[0]
            if dtype==T.FLOAT16:
                new_nodes.append(node("Cast",[x],[prefix+"x"],to=T.FLOAT));x=prefix+"x"
            new_nodes.extend([
                node("DynamicQuantizeLinear",[x],[prefix+"q",prefix+"scale",prefix+"zero"]),
                node("MatMulInteger",[prefix+"q",prefix+"weight",prefix+"zero",prefix+"wzero"],[prefix+"int"]),
                node("Mul",[prefix+"scale",prefix+"wscale"],[prefix+"scale_product"]),
                node("Cast",[prefix+"int"],[prefix+"float"],to=T.FLOAT),
                node("Mul",[prefix+"float",prefix+"scale_product"],[prefix+"result"] if dtype==T.FLOAT16 else list(n.output))
            ])
            # Both Cast nodes have distinct names; preserve original result
            # dtype and the existing downstream bias/activation operations.
            if dtype==T.FLOAT16:
                new_nodes[-2].name=prefix+"CastInteger"
                new_nodes[-1].name=prefix+"Rescale"
                new_nodes.append(h.make_node("Cast",[prefix+"result"],list(n.output),name=prefix+"CastOutput",to=T.FLOAT16))
            else:new_nodes[-1].name=prefix+"Rescale"
            stats.append({"node":n.name,"shape":list(shape),"weight":t.name,"int8_bytes":q.nbytes})
    # Remove only genuinely dead weight-transform chains / initializers.
    # deps includes references captured by nested ONNX subgraphs.
    needed={o.name for o in g.output};kept=[]
    for n in reversed(new_nodes):
        if needed.intersection(n.output):
            kept.append(n);needed.update(deps(n))
    live=set(needed)|{o for n in kept for o in n.output}
    g.ClearField("node");g.node.extend(reversed(kept))
    retained=[t for t in g.initializer if t.name in needed]
    g.ClearField("initializer");g.initializer.extend(retained+new_weights)
    stale=[v for v in g.value_info if v.name not in live]
    for v in stale:g.value_info.remove(v)
    return {"selected":len(stats),"counts":{str(k):v for k,v in counts.items()},
            "int8_weight_bytes":weight_file.stat().st_size,"matrices":stats}

def create(directory,profile):
    directory=Path(directory)
    selected={"all":SHAPES,"expand":SHAPES[:1],"contract":SHAPES[1:]}[profile]
    model=onnx.load(str(directory/GRAPH),load_external_data=False)
    target=directory/("decoder.ffn-dyn8-"+profile+".onnx")
    weights=directory/("decoder.ffn-dyn8-"+profile+".weights")
    report=rewrite(model,directory,weights,selected)
    onnx.save_model(model,str(target))
    onnx.checker.check_model(str(target))
    return target,weights,report

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory");p.add_argument("--profile",choices=("all","expand","contract"),default="all")
    a=p.parse_args();target,weights,report=create(a.directory,a.profile)
    print(json.dumps(report,indent=2))
