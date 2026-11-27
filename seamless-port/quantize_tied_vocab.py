"""Quantize the tied embedding/projection without deleting any token.

A single UINT8 table feeds both native MatMulNBits and a row-selective
embedding lookup. Only selected embedding rows are dequantized. No full
floating-point table is reconstructed in the ONNX graph.
"""
import copy,json,os
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h,numpy_helper as nh,TensorProto as T
from prune_english_vocab import GRAPH,attr,location,external,refs
from model_pack import tensors
from split_source_projections import deps

def quantize_rows(weight,group):
    a=np.asarray(weight,dtype=np.float32)
    if a.ndim!=2 or a.shape[1]%group or not np.isfinite(a).all():
        raise ValueError("invalid embedding weights")
    blocks=a.reshape(a.shape[0],-1,group)
    scale=np.max(np.abs(blocks),axis=-1)/np.float32(127)
    scale=np.where(scale>0,scale,np.float32(1)).astype(np.float32)
    q=(np.clip(np.rint(blocks/scale[...,None]),-127,127).astype(np.int16)+128).astype(np.uint8)
    return q,scale

def rewrite(model,vocab,group):
    g=model.graph
    weights=[t for t in g.initializer if vocab in t.dims]
    if len(weights)!=1:raise ValueError("expected exactly one tied vocabulary matrix")
    weight=weights[0]
    if len(weight.dims)!=2 or weight.dims[0]!=vocab or weight.dims[1]%group:
        raise ValueError("unexpected tied weight shape")
    if weight.data_type not in (T.FLOAT,T.FLOAT16):raise ValueError("unexpected weight dtype")
    dim=weight.dims[1]
    consumers={}
    for i,n in enumerate(g.node):
        for j,name in enumerate(n.input):consumers.setdefault(name,[]).append((i,j))
        if deps(n)-set(n.input) and weight.name in deps(n)-set(n.input):
            raise ValueError("nested reference to tied weight")
    embed=[];proj=[];chain=set()
    def follow(name,axis,dtype):
        uses=consumers.get(name,[])
        if not uses:raise ValueError("unused weight transform")
        for i,j in uses:
            n=g.node[i]
            if n.op_type=="Gather" and j==0 and axis==0 and attr(n,"axis",0)==0 and n.input[1]=="tokens":
                embed.append((i,dtype))
            elif n.op_type in ("Cast","Identity","Transpose") and j==0:
                chain.add(i)
                if n.op_type=="Cast":new_dtype=attr(n,"to")
                else:new_dtype=dtype
                new_axis=list(attr(n,"perm",[1,0])).index(axis) if n.op_type=="Transpose" else axis
                follow(n.output[0],new_axis,new_dtype)
            elif n.op_type=="MatMul" and j==1 and axis==1:proj.append((i,dtype))
            else:raise ValueError("unsupported tied weight consumer: "+n.name+" "+n.op_type)
    follow(weight.name,0,weight.data_type)
    if len(embed)!=1 or len(proj)!=1:raise ValueError("expected one embedding and one MatMul projection")
    ei,ed=embed[0];pi,pd=proj[0]
    if ed not in (T.FLOAT,T.FLOAT16) or pd not in (T.FLOAT,T.FLOAT16):
        raise ValueError("unsupported activation types")
    removed={weight.name}|{o for i in chain for o in g.node[i].output}
    for i,n in enumerate(g.node):
        if i not in chain|{ei,pi} and deps(n)&removed:
            raise ValueError("shared or nested weight transform")
    if removed&{x.name for x in g.output}:raise ValueError("weight is a graph output")
    prefix="tied8_"
    names={x.name for x in g.initializer}|{x.name for x in g.input}|{o for n in g.node for o in n.output}
    if any(x.startswith(prefix) for x in names):raise ValueError("name collision")
    # ONNX tensor names and node names are independent namespaces. In the
    # real FP16 graph an existing Cast node has the name of the preceding
    # MatMul output. Never derive a new node name from that output.
    node_names={n.name for n in g.node if n.name}
    def unique_node(label):
        name="tied8_node_"+label
        while name in node_names:name+="_"
        node_names.add(name)
        return name
    def n(op,inputs,output,**kw):
        return h.make_node(op,inputs,[output],name=unique_node(str(len(node_names))),**kw)
    e=g.node[ei];p=g.node[pi]
    en=[
        n("Gather",["tied8_q","tokens"],"tied8_rows",axis=0),
        n("Cast",["tied8_rows"],"tied8_rows_float",to=T.FLOAT),
        n("Sub",["tied8_rows_float","tied8_midpoint"],"tied8_centered"),
        n("Reshape",["tied8_scales","tied8_scale_shape"],"tied8_scale_rows"),
        n("Gather",["tied8_scale_rows","tokens"],"tied8_selected_scales",axis=0),
        n("Unsqueeze",["tied8_selected_scales","tied8_last_axis"],"tied8_broadcast"),
        n("Mul",["tied8_centered","tied8_broadcast"],"tied8_dequantized"),
        n("Reshape",["tied8_dequantized","tied8_embed_shape"],"tied8_embeddings"),
        n("Cast",["tied8_embeddings"],e.output[0],to=ed)]
    pn=[
        n("Cast",[p.input[0]],"tied8_projection_input",to=T.FLOAT),
        h.make_node("MatMulNBits",["tied8_projection_input","tied8_q","tied8_scales"],
                    ["tied8_projection_float"],name=unique_node("projection"),domain="com.microsoft",
                    K=dim,N=vocab,bits=8,block_size=group,accuracy_level=0),
        n("Cast",["tied8_projection_float"],p.output[0],to=pd)]
    nodes=[]
    for i,node in enumerate(g.node):
        if i==ei:nodes.extend(en)
        elif i==pi:nodes.extend(pn)
        elif i not in chain:nodes.append(node)
    named=[n.name for n in nodes if n.name]
    if len(named)!=len(set(named)):raise ValueError("duplicate node names after tied rewrite")
    g.ClearField("node");g.node.extend(nodes)
    original=copy.deepcopy(weight)
    g.initializer.remove(weight)
    g.initializer.extend([
        nh.from_array(np.array(128,np.float32),name="tied8_midpoint"),
        nh.from_array(np.array([vocab,dim//group],np.int64),name="tied8_scale_shape"),
        nh.from_array(np.array([-1],np.int64),name="tied8_last_axis"),
        nh.from_array(np.array([0,0,dim],np.int64),name="tied8_embed_shape")])
    for v in list(g.value_info):
        if v.name in removed:g.value_info.remove(v)
    if not any(x.domain=="com.microsoft" for x in model.opset_import):
        model.opset_import.append(h.make_opsetid("com.microsoft",1))
    return original

def main(source,dest,group=64):
    source,dest=Path(source),Path(dest)
    dest.mkdir(parents=True,exist_ok=False)
    meta=json.loads((source/"tokenizer.json").read_text())
    vocab=len(meta["table"])
    model=onnx.load(str(source/GRAPH),load_external_data=False)
    original=rewrite(model,vocab,group)
    # Read only the one 2-D tied matrix; batch quantization avoids large
    # temporary full-vocabulary FP32 arrays.
    weight=nh.to_array(original,base_dir=str(source))
    dim=weight.shape[1]
    filename="decoder.tied8.weights"
    scales=np.empty((vocab,dim//group),np.float32)
    with (dest/filename).open("wb") as out:
        for start in range(0,vocab,2048):
            q,s=quantize_rows(weight[start:start+2048],group)
            out.write(q.tobytes());scales[start:start+len(s)]=s
        qbytes=out.tell()
        out.write(scales.tobytes())
        sbytes=scales.nbytes
        del weight,scales,q,s
        seen={}
        for t in tensors(model.graph):
            if t.data_location!=T.EXTERNAL:continue
            p,off,length=location(t,source)
            key=(p,off,length)
            if key not in seen:
                seen[key]=out.tell()
                with p.open("rb") as f:
                    f.seek(off);remaining=length
                    while remaining:
                        chunk=f.read(min(remaining,4*1024*1024))
                        if not chunk:raise IOError("truncated external tensor")
                        out.write(chunk);remaining-=len(chunk)
            external(t,filename,seen[key],length)
        for name,dtype,shape,offset,size in (
            ("tied8_q",T.UINT8,[vocab,dim//group,group],0,qbytes),
            ("tied8_scales",T.FLOAT,[vocab*(dim//group)],qbytes,sbytes)):
            t=T(name=name,data_type=dtype,dims=shape);external(t,filename,offset,size)
            model.graph.initializer.append(t)
    onnx.save_model(model,str(dest/GRAPH))
    onnx.checker.check_model(str(dest/GRAPH))
    old=refs(source/GRAPH)
    others=set().union(*(refs(source/n) for n in ("encoder.int8.onnx","units.onnx","vocoder.onnx")))
    skip={GRAPH,"sha256.json","speech-smoke.json","decoder.incremental.fp16.validated.json"}|(old-others)
    for p in source.iterdir():
        if p.is_file() and p.name not in skip:os.link(p,dest/p.name)
    report={"vocabulary":vocab,"group_size":group,"matrix_bytes":qbytes+sbytes,
            "original_decoder_bytes":(source/GRAPH).stat().st_size+sum((source/n).stat().st_size for n in old),
            "candidate_decoder_bytes":(dest/GRAPH).stat().st_size+(dest/filename).stat().st_size}
    (dest/"tied8-conversion.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("source");p.add_argument("dest")
    p.add_argument("--group",type=int,choices=(16,32,64,128),default=64)
    a=p.parse_args();main(a.source,a.dest,a.group)
