"""Build a storage/CPU-compatibility FP16 candidate for FP32 acoustic graphs."""
import json
from pathlib import Path
import time
import numpy as np
import onnx
import onnxruntime as ort
from onnxconverter_common.float16 import convert_float_to_float16


def nrms(actual,expected):
    a=np.asarray(actual,dtype=np.float64);e=np.asarray(expected,dtype=np.float64)
    return float(np.sqrt(np.mean((a-e)**2))/max(float(np.sqrt(np.mean(e**2))),1e-6))


def main(component,directory):
    if component not in ("units","vocoder"):raise ValueError("FP16 experiment is only for units/vocoder")
    directory=Path(directory);source=directory/f"{component}.onnx"
    output=directory/f"{component}.fp16.onnx";marker=directory/f"{component}.fp16.validated.json"
    marker.unlink(missing_ok=True)
    result={"component":component,"precision":"fp16-candidate","parity":"PENDING","tests":[]}
    try:
        if not (directory/f"{component}.validated.json").exists():raise RuntimeError("validated FP32 export is required")
        fixtures=sorted(directory.glob(f"{component}.case*.npz"))
        if not fixtures:raise RuntimeError("independent FP32 fixtures are missing")
        base_graph=onnx.load(str(source),load_external_data=False)
        base_locations=set()
        for t in base_graph.graph.initializer:
            for e in t.external_data:
                if e.key=="location":base_locations.add(e.value)
        result["fp32_bytes"]=source.stat().st_size+sum((directory/x).stat().st_size for x in base_locations)
        del base_graph
        model=onnx.load(str(source),load_external_data=True)
        # Preserve fused scaled-dot-product attention subgraphs in FP32. The
        # generic converter otherwise half-converts one MatMul operand while
        # leaving its paired operand FP32, which makes the ONNX graph invalid.
        blocked=[n.name for n in model.graph.node if "/sdpa/" in n.name.lower()]
        result["blocked_sdpa_nodes"]=len(blocked)
        start=time.monotonic()
        model=convert_float_to_float16(model,keep_io_types=True,disable_shape_infer=True,
                                       node_block_list=blocked)
        onnx.save_model(model,str(output),save_as_external_data=True,all_tensors_to_one_file=True,
            location=output.name+".weights",size_threshold=1024)
        onnx.checker.check_model(str(output))
        result["conversion_seconds"]=time.monotonic()-start
        weights=directory/(output.name+".weights")
        result["fp16_bytes"]=output.stat().st_size+(weights.stat().st_size if weights.exists() else 0)
        options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
        session=ort.InferenceSession(str(output),sess_options=options,providers=["CPUExecutionProvider"])
        failures=[]
        for path in fixtures:
            with np.load(path,allow_pickle=False) as f:
                inputs={i.name:f["input_"+i.name] for i in session.get_inputs()}
                names=[o.name for o in session.get_outputs()]
                values=session.run(names,inputs);metrics={"fixture":path.name}
                for name,actual in zip(names,values):
                    expected=f["output_"+name]
                    error=nrms(actual,expected);metrics[name+"_nrms"]=error
                    if actual.shape!=expected.shape or not np.isfinite(actual).all():
                        failures.append(path.name+": invalid "+name+" output")
                    elif component=="units" and name=="durations":
                        if not np.array_equal(actual,expected):failures.append(path.name+": duration mismatch")
                    elif component=="units" and name=="logits":
                        agreement=float(np.mean(actual.argmax(-1)==expected.argmax(-1)))
                        metrics["unit_argmax_agreement"]=agreement
                        if error>0.10 or agreement!=1.0:failures.append(path.name+": unit logits/argmax mismatch")
                    elif component=="vocoder":
                        metrics[name+"_max_abs"]=float(np.max(np.abs(actual-expected)))
                        if error>0.02 or metrics[name+"_max_abs"]>0.05:failures.append(path.name+": waveform tolerance exceeded")
                result["tests"].append(metrics);print(json.dumps(metrics),flush=True)
        result["failures"]=failures
        result["parity"]="FAIL" if failures else "PASS"
        result["size_reduction_percent"]=100.0*(1-result["fp16_bytes"]/max(result["fp32_bytes"],1))
    except Exception as exc:
        result["parity"]="FAIL";result["reason"]=str(exc)
    marker.write_text(json.dumps(result,indent=2)+"\n")
    print(marker);print(json.dumps(result,indent=2),flush=True)
    if result["parity"]!="PASS":raise RuntimeError(result.get("reason") or "; ".join(result.get("failures",[])))

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("component",choices=("units","vocoder"));p.add_argument("directory",type=Path)
    a=p.parse_args();main(a.component,a.directory)
