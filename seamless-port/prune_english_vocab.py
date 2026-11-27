"""Physically shrink the embedding/output projection, retain the original ONNX IDs.

Original tokens map to compact embedding rows. Compact logits scatter into
the original vocabulary, using a finite negative sentinel for excluded IDs.
No change to the Java tokenizer or UnitY2 interface. Only experimental.
"""
import copy,json,os,re,shutil
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h,numpy_helper as nh,TensorProto as T
from model_pack import tensors

GRAPH="decoder.incremental.fp16.onnx"
def attr(n,key,default=None):
    return next((h.get_attribute_value(a) for a in n.attribute if a.name==key),default)

def transform(model,kept,vocab,unk):
    g=model.graph
    kept=np.asarray(kept,dtype=np.int64)
    if kept.ndim!=1 or len(kept)==0 or len(kept)>=vocab or np.any(np.diff(kept)<=0) or kept[0]<0 or kept[-1]>=vocab or unk not in kept:
        raise ValueError("invalid retained token IDs")
    axes={}
    for t in g.initializer:
        shape=list(t.dims)
        if vocab not in shape:continue
        if len(shape)!=2 or shape.count(vocab)!=1 or t.data_type not in (T.FLOAT,T.FLOAT16):
            raise ValueError("unsupported vocabulary tensor "+t.name+str(shape))
        axes[t.name]=shape.index(vocab)
    if not axes:raise ValueError("no vocabulary matrices found")
    # Validate every use of vocabulary matrices, including the tied weight
    # case where Gather and Transpose->MatMul consume one initializer.
    consumers={}
    for n in g.node:
        for i,x in enumerate(n.input):consumers.setdefault(x,[]).append((n,i))
    embed=[];project=[]
    def follow(name,axis,seen):
        if name in seen:raise ValueError("cyclic weight use")
        if not consumers.get(name):raise ValueError("unused vocabulary tensor "+name)
        for n,i in consumers[name]:
            if n.op_type=="Gather" and i==0 and axis==0 and attr(n,"axis",0)==0 and n.input[1]=="tokens":
                embed.append(n)
            elif n.op_type in ("Cast","Identity","Transpose") and i==0:
                perm=attr(n,"perm",[1,0]) if n.op_type=="Transpose" else [0,1]
                follow(n.output[0],list(perm).index(axis),seen|{name})
            elif n.op_type=="MatMul" and i==1 and axis==1:
                project.append(n)
            elif n.op_type=="Gemm" and i==1 and axis==(0 if attr(n,"transB",0) else 1):
                if len(n.input)>2 and n.input[2]:raise ValueError("projection bias not supported")
                project.append(n)
            else:raise ValueError("unhandled vocabulary weight use "+name+" -> "+n.op_type+" "+n.name)
    for name,axis in axes.items():follow(name,axis,set())
    if len(embed)!=1 or len(project)!=1:raise ValueError(f"expected one embedding and projection, got {len(embed)}/{len(project)}")
    # Verify the projection output reaches logits through only shape-preserving
    # floating casts/identity. This rejects hidden vocabulary dependencies.
    value=project[0].output[0]
    while value!="logits":
        uses=consumers.get(value,[])
        if len(uses)!=1 or uses[0][0].op_type not in ("Cast","Identity"):
            raise ValueError("unexpected logits path "+value)
        value=uses[0][0].output[0]
    logits=next(x for x in g.output if x.name=="logits")
    if logits.type.tensor_type.elem_type!=T.FLOAT:raise ValueError("expected FP32 logits output")
    mapping=np.full(vocab,int(np.searchsorted(kept,unk)),np.int64)
    mapping[kept]=np.arange(len(kept),dtype=np.int64)
    reserved={"english_id_map","english_compact_tokens","english_compact_logits","english_logit_indices","english_logit_base"}
    all_names={t.name for t in g.initializer}|{x.name for x in g.input}|{o for n in g.node for o in n.output}
    if reserved&all_names:raise ValueError("pruning node name collision")
    embed[0].input[1]="english_compact_tokens"
    producer=next(n for n in g.node if "logits" in n.output)
    producer.output[list(producer.output).index("logits")]="english_compact_logits"
    g.node.insert(0,h.make_node("Gather",["english_id_map","tokens"],["english_compact_tokens"],axis=0,name="EnglishVocabRemap"))
    g.node.append(h.make_node("ScatterElements",["english_logit_base","english_logit_indices","english_compact_logits"],["logits"],axis=2,name="EnglishVocabRestore"))
    g.initializer.extend([
        nh.from_array(mapping,name="english_id_map"),
        nh.from_array(kept.reshape(1,1,-1),name="english_logit_indices"),
        nh.from_array(np.full((1,1,vocab),np.finfo(np.float32).min,dtype=np.float32),name="english_logit_base")])
    # Existing inferred metadata for resized internal tensors is stale.
    stale=[v for v in g.value_info if any(d.HasField("dim_value") and d.dim_value==vocab for d in v.type.tensor_type.shape.dim)]
    for v in stale:g.value_info.remove(v)
    return axes

def location(t,root):
    entries={e.key:e.value for e in t.external_data}
    name=entries["location"]
    if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,240}",name) or name in (".",".."):raise ValueError("unsafe external file")
    p=root/name
    offset=int(entries.get("offset",0))
    size=int(entries.get("length",p.stat().st_size-offset))
    if offset<0 or size<0 or offset+size>p.stat().st_size:raise ValueError("bad external span")
    return p,offset,size

def external(t,filename,offset,size):
    t.ClearField("raw_data");t.ClearField("external_data");t.data_location=T.EXTERNAL
    for k,v in (("location",filename),("offset",str(offset)),("length",str(size))):
        e=t.external_data.add();e.key=k;e.value=v

def compact(model,source,dest,axes,kept):
    name="decoder.english.weights"
    seen={}
    with (dest/name).open("wb") as out:
        for t in tensors(model.graph):
            if t.name in axes:
                a=nh.to_array(copy.deepcopy(t),base_dir=str(source))
                a=np.ascontiguousarray(np.take(a,kept,axis=axes[t.name]))
                t.ClearField("dims");t.dims.extend(a.shape)
                off=out.tell();out.write(a.tobytes());external(t,name,off,a.nbytes)
                print("pruned",t.name,list(a.shape),flush=True)
            elif t.data_location==T.EXTERNAL:
                p,off,n=location(t,source)
                key=(p,off,n)
                if key not in seen:
                    seen[key]=out.tell()
                    with p.open("rb") as f:
                        f.seek(off);remaining=n
                        while remaining:
                            b=f.read(min(remaining,4*1024*1024))
                            if not b:raise IOError("truncated external tensor")
                            out.write(b);remaining-=len(b)
                external(t,name,seen[key],n)
    return (dest/name).stat().st_size

def refs(graph):
    m=onnx.load(str(graph),load_external_data=False)
    return {e.value for t in tensors(m.graph) for e in t.external_data if e.key=="location"}

def main(source,dest):
    source,dest=Path(source),Path(dest)
    dest.mkdir(parents=True,exist_ok=False)
    meta=json.loads((source/"tokenizer.json").read_text())
    plan=json.loads((source/"english-vocab.json").read_text())
    kept=plan["retained_ids"];vocab=len(meta["table"])
    if vocab!=plan["original_size"]:raise ValueError("vocabulary mismatch")
    if not set(meta["prefix"]+[meta[k] for k in ("eos","pad","unk","comma")]).issubset(kept):
        raise ValueError("essential IDs missing")
    model=onnx.load(str(source/GRAPH),load_external_data=False)
    old_refs=refs(source/GRAPH)
    other_refs=set().union(*(refs(source/n) for n in ("encoder.int8.onnx","units.onnx","vocoder.onnx")))
    old_bytes=(source/GRAPH).stat().st_size+sum((source/n).stat().st_size for n in old_refs)
    axes=transform(model,kept,vocab,meta["unk"])
    new_bytes=compact(model,source,dest,axes,kept)
    onnx.save_model(model,str(dest/GRAPH))
    onnx.checker.check_model(str(dest/GRAPH))
    skip={GRAPH,"sha256.json","speech-smoke.json"}|(old_refs-other_refs)
    for p in source.iterdir():
        if p.is_file() and p.name not in skip:os.link(p,dest/p.name)
    report={"original_size":vocab,"retained":len(kept),"pruned_tensors":axes,
            "original_decoder_bytes":old_bytes,"candidate_decoder_bytes":new_bytes+(dest/GRAPH).stat().st_size}
    (dest/"english-vocab-conversion.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)

if __name__=="__main__":
    import sys
    main(*sys.argv[1:])
