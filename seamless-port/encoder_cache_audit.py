"""Real encoder layerwise cache feasibility and operator-cost audit.

Expose existing Shaw attention tensors, without changing weights or arithmetic.
Measure Q, K, V stability and probability assigned to appended source frames.
These are necessary (not sufficient) conditions for exact incremental reuse.
Never promote a cache or a model from this diagnostic.
"""
import argparse
import collections
import copy
import json
from pathlib import Path
import re
import statistics
import time
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper as h, TensorProto as T

def attention_plan(model,expected_layers=24):
    g=model.graph
    nodes=list(g.node)
    consumers=collections.defaultdict(list)
    for n in nodes:
        for x in n.input:consumers[x].append(n)
    plan=[]
    for n in nodes:
        match=re.search(r"/speech_encoder/inner/layers\.(\d+)/self_attn/sdpa/MatMul_1$",n.name)
        if not match or n.op_type!="MatMul":continue
        layer=int(match.group(1));prefix=n.name.rsplit("/",1)[0]+"/"
        soft=[s for s in nodes if s.name.startswith(prefix) and s.op_type=="Softmax"]
        if len(soft)!=1:raise ValueError(f"layer {layer}: expected one Shaw Softmax, got {len(soft)}")
        # Find attention value multiplication through the existing Cast/Identity
        # chain. Fail closed if graph differs; do not infer arbitrary tensors.
        pending=[soft[0].output[0]];visited=set();value=[]
        while pending:
            x=pending.pop()
            if x in visited:continue
            visited.add(x)
            for s in consumers[x]:
                if not s.name.startswith(prefix):continue
                if s.op_type in ("Cast","Identity") and len(s.input)==1:pending.extend(s.output)
                elif s.op_type=="MatMul" and s.input[0]==x:value.append(s)
        if len(value)!=1 or len(n.input)!=2:raise ValueError(f"layer {layer}: unexpected attention dataflow")
        # Q:[B,H,T,D], transposed K:[B,H,D,T], V:[B,H,T,D],
        # probabilities:[B,H,T,T], attention result:[B,H,T,D].
        plan.append({"layer":layer,"q":n.input[0],"kt":n.input[1],
                     "v":value[0].input[1],"prob":soft[0].output[0],
                     "context":value[0].output[0]})
    plan.sort(key=lambda x:x["layer"])
    if [x["layer"] for x in plan]!=list(range(expected_layers)):
        raise ValueError("unexpected Shaw layer inventory: "+str([x["layer"] for x in plan]))
    return plan

def expose(model,plan):
    # This is the existing INT8 encoder with FP32 activations, not a new export.
    # Assert runtime types and shapes below; ORT rejects mismatched output types.
    m=copy.deepcopy(model)
    existing={v.name for v in m.graph.output}
    for layer in plan:
        for key in ("q","kt","v","prob","context"):
            name=layer[key]
            if name not in existing:
                m.graph.output.append(h.make_tensor_value_info(name,T.FLOAT,[None]*4))
                existing.add(name)
    return m

def drift(previous,current,axis,atol=1e-6,rtol=1e-5):
    if previous.ndim!=current.ndim or any(a!=b for i,(a,b) in enumerate(zip(previous.shape,current.shape)) if i!=axis):
        raise ValueError("incompatible cached state shapes")
    old=previous.shape[axis]
    if current.shape[axis]<old or not old:raise ValueError("source state shrank or empty")
    a=np.moveaxis(previous,axis,0).reshape(old,-1)
    b=np.moveaxis(current,axis,0)[:old].reshape(old,-1)
    d=b.astype(np.float64)-a.astype(np.float64)
    exact=np.all(a==b,axis=1)
    near=np.all(np.abs(d)<=atol+rtol*np.abs(a),axis=1)
    def prefix(mask):
        bad=np.flatnonzero(~mask)
        return int(bad[0]) if len(bad) else len(mask)
    return {"old_rows":old,"exact_prefix":prefix(exact),"tolerance_prefix":prefix(near),
            "exact_rows":int(exact.sum()),"nrms":float(np.linalg.norm(d)/max(np.linalg.norm(a.astype(np.float64)),1e-12)),
            "max_abs":float(np.max(np.abs(d)))}

def future_mass(prob,old):
    if prob.ndim!=4 or prob.shape[2]!=prob.shape[3] or not 0<old<prob.shape[2]:
        raise ValueError("invalid attention matrix or append boundary")
    if np.min(prob)<-1e-6 or not np.allclose(prob.sum(-1),1,rtol=1e-4,atol=1e-5):
        raise ValueError("not normalized attention probabilities")
    mass=prob[:,:,:old,old:].sum(axis=-1)
    worst=mass.max(axis=(0,1))
    active=np.flatnonzero(worst>1e-7)
    return {"old_queries":old,"new_keys":prob.shape[3]-old,
            "max_mass":float(worst.max()),"mean_mass":float(mass.mean()),
            "first_affected_query":int(active[0]) if len(active) else None,
            "queries_with_new_key_mass_gt_1e-7":int(len(active))}

def summarize_profile(path):
    events=json.loads(Path(path).read_text())
    groups=collections.defaultdict(lambda:{"duration_us":0.0,"calls":0})
    nodes=collections.defaultdict(lambda:{"duration_us":0.0,"calls":0})
    for e in events:
        if e.get("cat")!="Node" or not e.get("name","").endswith("_kernel_time"):continue
        a=e.get("args",{});op=a.get("op_name","unknown");provider=a.get("provider","unknown")
        for row in (groups[provider+"/"+op],nodes[e["name"]]):
            row["duration_us"]+=float(e.get("dur",0));row["calls"]+=1
    if not groups:raise ValueError("ORT profile did not contain kernel timings")
    ordered=lambda d: sorted(({"name":k,**v} for k,v in d.items()),key=lambda x:x["duration_us"],reverse=True)
    return {"operator_groups":ordered(groups),"top_nodes":ordered(nodes)[:40],
            "note":"Kernel event durations include profiling overhead and are not end-to-end wall time."}

def run(directory,features,output):
    directory=Path(directory);out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True)
    path=directory/"encoder.int8.onnx"
    model=onnx.load(str(path),load_external_data=False)
    plan=attention_plan(model)
    f=np.fromfile(features,dtype="<f4").reshape(-1,80)
    # Use the full same fixture and three adjacent 64-frame update pairs.
    end=len(f)//64*64
    if end<512:raise ValueError("fixture too short")
    pairs=sorted(set((a,a+64) for a in (192,(end//128)*64,end-64)))
    lengths=sorted(set(v for pair in pairs for v in pair))
    report={"scope":"Real native ARM64 encoder diagnostic; not a cache implementation or phone timing.",
            "input_frames":len(f),"pairs":pairs,"attention_plan":plan,"baseline":[],
            "instrumented":{},"comparisons":[]}
    def save():
        (out/"encoder-cache-audit.json").write_text(json.dumps(report,indent=2)+"\n")
    save()
    def opts():
        o=ort.SessionOptions();o.intra_op_num_threads=4;o.inter_op_num_threads=1
        return o
    def feed(n):return {"fbank":np.ascontiguousarray(f[None,:n,:])}
    # Uninstrumented wall-clock timing. Profiling runs are separate because
    # recording events and retaining graph outputs may perturb optimization.
    baseline=ort.InferenceSession(str(path),opts(),providers=["CPUExecutionProvider"])
    refs={}
    for n in lengths:
        samples=[]
        for iteration in range(3):
            t=time.monotonic();result=baseline.run(["encoded"],feed(n))[0];elapsed=time.monotonic()-t
            if not np.isfinite(result).all():raise ValueError("non-finite encoder output")
            if iteration:samples.append(elapsed)
        refs[n]=result
        report["baseline"].append({"frames":n,"encoded_frames":result.shape[1],"median_seconds":statistics.median(samples),"samples":samples})
    del baseline
    save()
    o=opts();o.enable_profiling=True;o.profile_file_prefix=str(out/"ort-encoder-profile")
    prof=ort.InferenceSession(str(path),o,providers=["CPUExecutionProvider"])
    prof.run(["encoded"],feed(lengths[0]))
    for n in (lengths[0],lengths[len(lengths)//2],lengths[-1]):prof.run(["encoded"],feed(n))
    profile=prof.end_profiling();del prof
    report["profile"]=summarize_profile(profile);save()
    # Save beside the original graph so all external initializer locations
    # remain correct. Delete temporary graph even on error.
    probe_path=directory/"encoder.cache-audit.onnx"
    if probe_path.exists():raise ValueError("temporary probe graph already exists")
    onnx.save_model(expose(model,plan),str(probe_path))
    try:
        onnx.checker.check_model(str(probe_path))
        session=ort.InferenceSession(str(probe_path),opts(),providers=["CPUExecutionProvider"])
        names=list(dict.fromkeys(["encoded"]+[l[k] for l in plan for k in ("q","kt","v","prob","context")]))
        def capture(n):
            values=dict(zip(names,session.run(names,feed(n))))
            ref=refs[n]
            if values["encoded"].shape!=ref.shape or not np.allclose(values["encoded"],ref,rtol=1e-4,atol=1e-5):
                raise ValueError(f"instrumentation perturbed encoder output at {n} frames")
            for name,a in values.items():
                if a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError("unexpected tensor: "+name)
            report["instrumented"][str(n)]={"encoded_shape":list(ref.shape),"parity":True}
            return values
        for a,b in pairs:
            prev,cur=capture(a),capture(b)
            layers=[]
            for l in plan:
                q0,q1=prev[l["q"]],cur[l["q"]]
                old=q0.shape[2];new=q1.shape[2]
                if new<=old or q0.ndim!=4 or q1.ndim!=4:raise ValueError("unexpected query time dimension")
                for data,t in ((prev,old),(cur,new)):
                    q,kt,v,p,c=(data[l[k]] for k in ("q","kt","v","prob","context"))
                    if kt.shape!=(q.shape[0],q.shape[1],q.shape[3],t) or v.shape!=q.shape or c.shape!=q.shape or p.shape!=(q.shape[0],q.shape[1],t,t):
                        raise ValueError("unexpected attention layout at layer "+str(l["layer"]))
                row={"layer":l["layer"],"new_attention_mass":future_mass(cur[l["prob"]],old)}
                for k,axis in (("q",2),("kt",3),("v",2),("context",2)):
                    row[k]=drift(prev[l[k]],cur[l[k]],axis)
                layers.append(row)
            pair={"frames_before":a,"frames_after":b,"encoded":drift(prev["encoded"],cur["encoded"],1),"layers":layers}
            report["comparisons"].append(pair);save()
            print(json.dumps({"frames":[a,b],"encoded":pair["encoded"],
                "layer_summary":[{"layer":l["layer"],"k_exact_prefix":l["kt"]["exact_prefix"],
                "v_exact_prefix":l["v"]["exact_prefix"],"context_exact_prefix":l["context"]["exact_prefix"],
                "new_key_mass":l["new_attention_mass"]["max_mass"]} for l in layers]}),flush=True)
            del prev,cur
        del session
        report["status"]="PASS"
        report["exact_cache_demonstrated"]=False
        save()
        print(json.dumps({"baseline":report["baseline"],"top_operators":report["profile"]["operator_groups"][:12]}),flush=True)
    finally:probe_path.unlink(missing_ok=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory");p.add_argument("features");p.add_argument("output")
    a=p.parse_args();run(a.directory,a.features,a.output)
