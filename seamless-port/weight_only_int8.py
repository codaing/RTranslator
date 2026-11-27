"""Storage INT8, FP32 activation and matrix arithmetic.

Group-wise symmetric weight quantization (16 input weights per output channel).
Uses standard ONNX DequantizeLinear/Reshape/Transpose, not an Android custom op.
ORT may expand/pack weights at session creation: compressed storage is not a
promise of compressed resident memory. The original model remains untouched.
"""
import math
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T
from onnx.external_data_helper import uses_external_data, set_external_data

GROUP=16


def tensors(graph):
    yield from graph.initializer
    for n in graph.node:
        for a in n.attribute:
            if a.type==onnx.AttributeProto.TENSOR:
                yield a.t
            elif a.type==onnx.AttributeProto.TENSORS:
                yield from a.tensors
            elif a.type==onnx.AttributeProto.GRAPH:
                yield from tensors(a.g)
            elif a.type==onnx.AttributeProto.GRAPHS:
                for g in a.graphs:yield from tensors(g)


def grouped_weight(weight,group=GROUP):
    """Return int8 blocks and per-block float32 scales for [in,out] weights."""
    if weight.ndim!=2 or weight.shape[0]%group:
        raise ValueError("weight input dimension must divide group size")
    blocks=np.ascontiguousarray(weight.T).reshape(-1,group)
    if not np.isfinite(blocks).all():
        raise ValueError("non-finite weights")
    scale=np.max(np.abs(blocks),axis=1)/np.float32(127)
    scale=np.where(scale>0,scale,np.float32(1)).astype(np.float32)
    quantized=np.clip(np.rint(blocks/scale[:,None]),-127,127).astype(np.int8)
    return quantized,scale


def convert(source,output,selected,group=GROUP,native=False,asymmetric=False):
    source,output=Path(source),Path(output)
    model=onnx.load(str(source),load_external_data=False)
    graph=model.graph
    if native and not any(o.domain=="com.microsoft" for o in model.opset_import):
        model.opset_import.append(h.make_opsetid("com.microsoft",1))
    initializers={t.name:t for t in graph.initializer}
    selected=set(selected)
    if not selected:raise ValueError("empty weight-only selection")
    names={v.name for v in list(graph.input)+list(graph.output)+list(graph.initializer)}
    names.update(o for n in graph.node for o in n.output)
    data_name=output.name+".weights"
    replacements={}
    packed=[]
    native_inputs={}
    # Store all new arrays and all retained external tensors in one compact file.
    # Otherwise a shared original FP32 external-data file could defeat compression.
    with (output.parent/data_name).open("wb") as data:
        def external(t,offset,length):
            t.raw_data=b""
            set_external_data(t,location=data_name,offset=offset,length=length)
            t.ClearField("raw_data")
            t.data_location=T.EXTERNAL
        def append_array(name,value):
            value=np.ascontiguousarray(value)
            offset=data.tell()
            data.write(memoryview(value).cast("B"))
            t=T(name=name,data_type=h.np_dtype_to_tensor_dtype(value.dtype),dims=list(value.shape))
            external(t,offset,value.nbytes)
            graph.initializer.append(t)
            packed.append(name)
        new_nodes=[]
        for n in graph.node:
            if n.name not in selected:
                new_nodes.append(n)
                continue
            if n.op_type!="MatMul" or n.input[1] not in initializers:
                raise RuntimeError("unexpected selected op: "+n.name)
            weight_name=n.input[1]
            if weight_name not in replacements:
                t=initializers[weight_name]
                weight=nh.to_array(t,base_dir=str(source.parent))
                if weight.dtype!=np.float32:raise ValueError("expected FP32 weight")
                tag="weight_only/"+str(len(replacements))
                if native:
                    k,n=weight.shape
                    if asymmetric:
                        blocks=np.ascontiguousarray(weight.T).reshape(-1,group)
                        lo=np.minimum(blocks.min(axis=1),0)
                        hi=np.maximum(blocks.max(axis=1),0)
                        s=(hi-lo)/np.float32(255)
                        s=np.where(s>0,s,np.float32(1)).astype(np.float32)
                        z=np.clip(np.rint(-lo/s),0,255).astype(np.uint8)
                        q=np.clip(np.rint(blocks/s[:,None])+z[:,None].astype(np.float32),0,255).astype(np.uint8)
                        del blocks,lo,hi
                    else:
                        q,s=grouped_weight(weight,group)
                        q=(q.astype(np.int16)+128).astype(np.uint8)
                        z=None
                    qn,sn,zn=[tag+x for x in ("/packed","/scale","/zero")]
                    if any(x in names for x in (qn,sn,zn)):raise ValueError("name collision")
                    append_array(qn,q.reshape(n,k//group,group))
                    append_array(sn,s.reshape(-1))
                    extra=[qn,sn]
                    if z is not None:
                        append_array(zn,z.reshape(-1));extra.append(zn)
                    native_inputs[weight_name]=(extra,k,n)
                    replacements[weight_name]=qn
                    del weight,q,s,z
                else:
                    q,s=grouped_weight(weight,group)
                    qn,sn,sh,dq,reshaped,restored=[tag+x for x in ("/q","/scale","/shape","/dequant","/rows","/weight")]
                    if any(x in names for x in (qn,sn,sh,dq,reshaped,restored)):raise ValueError("name collision")
                    append_array(qn,q)
                    append_array(sn,s)
                    graph.initializer.append(nh.from_array(np.array([weight.shape[1],weight.shape[0]],np.int64),name=sh))
                    new_nodes.extend([
                        h.make_node("DequantizeLinear",[qn,sn],[dq],axis=0,name=tag+"/DequantizeLinear"),
                        h.make_node("Reshape",[dq,sh],[reshaped],name=tag+"/Reshape"),
                        h.make_node("Transpose",[reshaped],[restored],perm=[1,0],name=tag+"/Transpose")])
                    replacements[weight_name]=restored
                    del weight,q,s
            if native:
                extra,k,nout=native_inputs[weight_name]
                new_nodes.append(h.make_node("MatMulNBits",[n.input[0]]+extra,list(n.output),
                    domain="com.microsoft",name=n.name,K=k,N=nout,bits=8,block_size=group,accuracy_level=0))
            else:
                n.input[1]=replacements[weight_name]
                new_nodes.append(n)
        del graph.node[:];graph.node.extend(new_nodes)
        # Retain originals that are used by any other operation or graph output.
        live={i for n in graph.node for i in n.input}|{v.name for v in graph.input}|{v.name for v in graph.output}
        nested=any(a.type in (onnx.AttributeProto.GRAPH,onnx.AttributeProto.GRAPHS) for n in graph.node for a in n.attribute)
        if not nested:
            kept=[t for t in graph.initializer if t.name not in replacements or t.name in live]
            del graph.initializer[:];graph.initializer.extend(kept)
        for t in tensors(graph):
            if not uses_external_data(t):continue
            info={e.key:e.value for e in t.external_data}
            if info["location"]==data_name:continue
            src=(source.parent/info["location"]).resolve()
            if not src.is_relative_to(source.parent.resolve()):raise ValueError("invalid external location")
            old_offset=int(info.get("offset","0"))
            length=int(info["length"]) if "length" in info else src.stat().st_size-old_offset
            if length<0 or old_offset<0 or old_offset+length>src.stat().st_size:raise ValueError("invalid external range")
            offset=data.tell()
            with src.open("rb") as f:
                f.seek(old_offset)
                remaining=length
                while remaining:
                    block=f.read(min(remaining,8*1024*1024))
                    if not block:raise EOFError(str(src))
                    data.write(block);remaining-=len(block)
            external(t,offset,length)
    if len(replacements)==0:raise RuntimeError("no weight-only quantization performed")
    del graph.value_info[:]
    onnx.save_model(model,str(output))
    onnx.checker.check_model(str(output))
    print(f"Weight-only INT8: {len(replacements)} weight tensors, group={group}, external bytes={(output.parent/data_name).stat().st_size}",flush=True)
    return len(replacements)
