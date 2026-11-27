"""Convert the incremental decoder to FP16 while retaining FP32 ONNX I/O/cache tensors."""
import json
from pathlib import Path
import time
import numpy as np
import onnx
import onnxruntime as ort
from onnxconverter_common.float16 import convert_float_to_float16,DEFAULT_OP_BLOCK_LIST
from fp16_candidate import fix_mixed_mul_types


def nrms(actual,expected):
    a=np.asarray(actual,dtype=np.float64);e=np.asarray(expected,dtype=np.float64)
    return float(np.sqrt(np.mean((a-e)**2))/max(float(np.sqrt(np.mean(e**2))),1e-6))


def walk(graph):
    for node in graph.node:
        yield node
        for attr in node.attribute:
            if attr.type==onnx.AttributeProto.GRAPH:yield from walk(attr.g)
            elif attr.type==onnx.AttributeProto.GRAPHS:
                for subgraph in attr.graphs:yield from walk(subgraph)


def main(directory):
    directory=Path(directory)
    source=directory/"decoder.incremental.fp32.onnx"
    output=directory/"decoder.incremental.fp16.onnx"
    marker=directory/"decoder.incremental.fp16.validated.json"
    marker.unlink(missing_ok=True)
    result={"component":"decoder","decoder_mode":"incremental-kv-cache",
            "precision":"fp16-incremental-experimental","parity":"PENDING","tests":[]}
    try:
        meta=json.loads((directory/"decoder.incremental.meta.json").read_text())
        baseline=json.loads((directory/"decoder.incremental.fp32.validated.json").read_text())
        if baseline.get("parity")!="PASS":raise RuntimeError("FP32 cache parity gate missing")
        base=onnx.load(str(source),load_external_data=False)
        locations={e.value for t in base.graph.initializer for e in t.external_data if e.key=="location"}
        result["fp32_bytes"]=source.stat().st_size+sum((directory/x).stat().st_size for x in locations)
        del base
        model=onnx.load(str(source),load_external_data=True)
        blocked=[n.name for n in walk(model.graph) if "/sdpa/" in n.name.lower()]
        result["blocked_sdpa_nodes"]=len(blocked)
        start=time.monotonic()
        model=convert_float_to_float16(model,keep_io_types=True,disable_shape_infer=True,
            op_block_list=list(DEFAULT_OP_BLOCK_LIST)+["Mul"],
            node_block_list=blocked)
        result["conversion_seconds"]=time.monotonic()-start
        result["mixed_type_casts"]=fix_mixed_mul_types(model.graph)
        weights=Path(str(output)+".weights")
        weights.unlink(missing_ok=True)
        onnx.save_model(model,str(output),save_as_external_data=True,all_tensors_to_one_file=True,
            location=output.name+".weights",size_threshold=1024)
        onnx.checker.check_model(str(output))
        result["fp16_bytes"]=output.stat().st_size+(weights.stat().st_size if weights.exists() else 0)
        result["size_reduction_percent"]=100.0*(1-result["fp16_bytes"]/max(result["fp32_bytes"],1))
        options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
        session=ort.InferenceSession(str(output),sess_options=options,providers=["CPUExecutionProvider"])
        failures=[];total_seconds=0.0
        for path in sorted(directory.glob("decoder.incremental.case*.npz")):
            with np.load(path,allow_pickle=False) as f:
                steps=int(f["steps"][0]);layers=int(meta["layers"]);heads=int(meta["heads"]);dim=int(meta["head_dim"])
                caches=[np.empty((1,heads,0,dim),dtype=np.float32) for _ in range(layers*2)]
                for step in range(steps):
                    feeds={"tokens":f[f"input_tokens_{step}"],"encoded":f["encoded"],
                           "position":f[f"input_position_{step}"]}
                    feeds.update({n:v for n,v in zip(meta["past_names"],caches)})
                    t0=time.monotonic();values=session.run(meta["output_names"],feeds)
                    elapsed=time.monotonic()-t0;total_seconds+=elapsed
                    actuals=values[:3]
                    expected=[f[f"expected_{name}_{step}"] for name in ("logits","features","p_choose")]
                    metrics={"fixture":path.name,"step":step,"seconds":elapsed}
                    for name,actual,reference in zip(("logits","features","p_choose"),actuals,expected):
                        error=nrms(actual,reference);metrics[name+"_nrms"]=error
                        if actual.shape!=reference.shape or not np.isfinite(actual).all() or error>0.10:
                            failures.append(f"{path.name} step {step}: {name} error {error:.3f}")
                    agreement=bool(np.array_equal(actuals[0].argmax(-1),expected[0].argmax(-1)))
                    decision=bool((np.min(actuals[2][...,-1,-1])>=0.5)==(np.min(expected[2][...,-1,-1])>=0.5))
                    metrics.update(top_token_agreement=agreement,read_write_agreement=decision)
                    if not agreement:failures.append(f"{path.name} step {step}: token mismatch")
                    if not decision:failures.append(f"{path.name} step {step}: READ/WRITE mismatch")
                    caches=values[3:]
                    result["tests"].append(metrics);print(json.dumps(metrics),flush=True)
        result["total_inference_seconds"]=total_seconds
        result["failures"]=failures
        result["parity"]="PASS" if not failures else "FAIL"
    except Exception as exc:
        result["parity"]="FAIL";result["reason"]=str(exc)
    marker.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2),flush=True)
    if result["parity"]!="PASS":raise RuntimeError(result.get("reason") or "; ".join(result.get("failures",[])))


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("directory",type=Path)
    a=p.parse_args();main(a.directory)
