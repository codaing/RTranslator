"""Try accuracy-gated native INT8 policies for the incremental cache graph."""
import gc
import json
from pathlib import Path
import time
import numpy as np
import onnx
import onnxruntime as ort
from weight_only_int8 import convert
from fold_weights import prepare
from decoder_session import configure_native


def nrms(actual,expected):
    a=np.asarray(actual,dtype=np.float64);e=np.asarray(expected,dtype=np.float64)
    return float(np.sqrt(np.mean((a-e)**2))/max(float(np.sqrt(np.mean(e**2))),1e-6))


def run_candidate(output,selected,group,asymmetric,meta,directory):
    for path in (output,Path(str(output)+".weights")):
        path.unlink(missing_ok=True)
    convert(prepared_graph,output,selected,group=group,native=True,asymmetric=asymmetric,bits=8)
    onnx.checker.check_model(str(output))
    options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
    configure_native(options)
    session=ort.InferenceSession(str(output),sess_options=options,providers=["CPUExecutionProvider"])
    output_names=meta["output_names"]
    failures=[];tests=[]
    for path in sorted(directory.glob("decoder.incremental.case*.npz")):
        with np.load(path,allow_pickle=False) as f:
            steps=int(f["steps"][0]);layers=int(meta["layers"]);heads=int(meta["heads"]);dim=int(meta["head_dim"])
            caches=[np.empty((1,heads,0,dim),dtype=np.float32) for _ in range(layers*2)]
            for step in range(steps):
                feeds={"tokens":f[f"input_tokens_{step}"],"encoded":f["encoded"],"position":f[f"input_position_{step}"]}
                feeds.update({n:v for n,v in zip(meta["past_names"],caches)})
                t0=time.monotonic();values=session.run(output_names,feeds);elapsed=time.monotonic()-t0
                actuals=values[:3]
                expected=[f[f"expected_{name}_{step}"] for name in ("logits","features","p_choose")]
                metrics={"fixture":path.name,"step":step,"seconds":elapsed}
                for name,actual,reference in zip(("logits","features","p_choose"),actuals,expected):
                    error=nrms(actual,reference);metrics[name+"_nrms"]=error
                    if actual.shape!=reference.shape or not np.isfinite(actual).all() or error>0.10:
                        failures.append(f"{path.name} step {step}: {name} parity failed ({error:.3f})")
                agreement=bool(np.array_equal(actuals[0].argmax(-1),expected[0].argmax(-1)))
                decision=bool((np.min(actuals[2][...,-1,-1])>=0.5)==(np.min(expected[2][...,-1,-1])>=0.5))
                metrics.update(top_token_agreement=agreement,read_write_agreement=decision)
                if not agreement:failures.append(f"{path.name} step {step}: top token differs")
                if not decision:failures.append(f"{path.name} step {step}: READ/WRITE decision differs")
                caches=values[3:]
                tests.append(metrics)
                print(json.dumps(metrics),flush=True)
    del session
    gc.collect()
    graph=onnx.load(str(output),load_external_data=False)
    quantized_ops=sum(n.op_type=="MatMulNBits" and n.domain=="com.microsoft" for n in graph.graph.node)
    del graph
    weights=Path(str(output)+".weights")
    total_bytes=output.stat().st_size+(weights.stat().st_size if weights.exists() else 0)
    return {"selected_nodes":len(selected),"quantized_ops":quantized_ops,"model_bytes":total_bytes,
            "tests":tests,"failures":failures,
            "parity":"PASS" if not failures and quantized_ops==len(selected) else "FAIL"}


def main(directory):
    global prepared_graph
    directory=Path(directory)
    source=directory/"decoder.incremental.fp32.onnx"
    output=directory/"decoder.incremental.native8.onnx"
    report=directory/"decoder.incremental.native8.report.json"
    marker=directory/"decoder.incremental.native8.validated.json"
    meta=json.loads((directory/"decoder.incremental.meta.json").read_text())
    baseline=json.loads((directory/"decoder.incremental.fp32.validated.json").read_text())
    if baseline.get("parity")!="PASS":raise RuntimeError("FP32 incremental parity gate missing")
    prepared_graph=prepare(source)
    original_locations=set()
    if prepared_graph!=source:
        original=onnx.load(str(source),load_external_data=False)
        original_locations={e.value for t in original.graph.initializer for e in t.external_data if e.key=="location"}
        del original
    graph=onnx.load(str(prepared_graph),load_external_data=False)
    initializers={t.name:t for t in graph.graph.initializer}
    shape_nodes={}
    shape_counts={}
    for node in graph.graph.node:
        if node.op_type!="MatMul" or len(node.input)<2 or node.input[1] not in initializers:continue
        shape=tuple(int(x) for x in initializers[node.input[1]].dims)
        shape_counts[str(shape)]=shape_counts.get(str(shape),0)+1
        if len(shape)==2 and shape[0]!=shape[1] and min(shape)>=512 and max(shape)<=16384 and all(x%16==0 for x in shape):
            shape_nodes.setdefault(shape,[]).append(node.name)
    del graph
    if sorted(len(v) for v in shape_nodes.values())!=[24,24]:
        raise RuntimeError(f"expected two groups of 24 FFN MatMuls; groups={ {str(k):len(v) for k,v in shape_nodes.items()} }, all_shapes={shape_counts}")
    all_nodes=shape_nodes[(1024,8192)]+shape_nodes[(8192,1024)]
    attempts=[
        ("all_asymmetric_group16",all_nodes,True),
        ("shape_1024x8192_asymmetric_group16",shape_nodes[(1024,8192)],True),
        ("shape_8192x1024_asymmetric_group16",shape_nodes[(8192,1024)],True),
        ("all_symmetric_group16",all_nodes,False),
    ]
    start=time.monotonic();attempt_reports=[];winner=None
    for policy,selected,asymmetric in attempts:
        print(f"Trying {policy}: {len(selected)} matrices",flush=True)
        result=run_candidate(output,selected,16,asymmetric,meta,directory)
        result.update(policy=policy,asymmetric=asymmetric,group_size=16)
        attempt_reports.append(result)
        if result["parity"]=="PASS":
            winner=result;break
        for path in (output,Path(str(output)+".weights")):path.unlink(missing_ok=True)
        gc.collect()
    prepared_locations=set()
    graph=onnx.load(str(prepared_graph),load_external_data=False)
    prepared_locations={e.value for t in graph.graph.initializer for e in t.external_data if e.key=="location"}
    del graph
    prepared_graph.unlink(missing_ok=True)
    source.unlink(missing_ok=True)
    for location in (prepared_locations|original_locations):(directory/location).unlink(missing_ok=True)
    result={"component":"decoder","precision":"int8-native-incremental-experimental",
        "decoder_mode":"incremental-kv-cache","shape_counts":shape_counts,"attempts":attempt_reports,
        "winner":winner,"parity":"PASS" if winner else "FAIL",
        "validation_seconds":time.monotonic()-start}
    report.write_text(json.dumps(result,indent=2)+"\n")
    if not winner:raise RuntimeError("no native INT8 incremental policy met parity gates")
    marker.write_text(json.dumps({**winner,"component":"decoder","precision":"int8-native-incremental-experimental",
        "decoder_mode":"incremental-kv-cache","shape_counts":shape_counts,
        "validation_seconds":time.monotonic()-start},indent=2)+"\n")
    print(marker)


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("directory",type=Path)
    a=p.parse_args();main(a.directory)
