"""Screen MNN on complete SeamlessStreaming stages, not isolated MatMuls.

Capture real inputs from the unchanged projected ONNX speech pipeline once.
Convert each stage independently; compare every output and measure dynamic
shape changes, data copies and warmed inference. No Android/GPU speed claim.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time
import traceback
import numpy as np

STAGES={"encoder":"encoder.int8.onnx","projection":"decoder.source-projections.onnx",
        "decoder":"decoder.cached-step.onnx","units":"units.onnx","vocoder":"vocoder.onnx"}

def capture(directory,features,out):
    import onnxruntime as ort
    from speech_smoke import run
    root=Path(out);root.mkdir(parents=True,exist_ok=True)
    real_session=ort.InferenceSession
    by_filename={v:k for k,v in STAGES.items()}
    class CaptureSession:
        def __init__(self,path,*args,**kw):
            self.inner=real_session(path,*args,**kw)
            self.stage=by_filename.get(Path(path).name);self.calls=0
        def __getattr__(self,name):return getattr(self.inner,name)
        def run(self,names,feeds,*args,**kw):
            result=self.inner.run(names,feeds,*args,**kw)
            self.calls+=1
            if self.stage and self.calls in (1,5,12,25,45):
                selected=names or [o.name for o in self.inner.get_outputs()]
                case=root/(self.stage+"-%03d.npz"%self.calls)
                arrays={f"in_{i}":v for i,v in enumerate(feeds.values())}
                arrays.update({f"out_{i}":v for i,v in enumerate(result)})
                np.savez(case,**arrays)
                case.with_suffix(".json").write_text(json.dumps({"inputs":list(feeds),"outputs":selected}))
            return result
    ort.InferenceSession=CaptureSession
    try:run(directory,features,str(root/"baseline.wav"),source_projections=True)
    finally:ort.InferenceSession=real_session
    (root/"baseline.json").write_text((Path(directory)/"speech-smoke.json").read_text())

def load_case(path):
    meta=json.loads(path.with_suffix(".json").read_text())
    with np.load(path,allow_pickle=False) as f:
        feeds={n:f[f"in_{i}"] for i,n in enumerate(meta["inputs"])}
        outputs={n:f[f"out_{i}"] for i,n in enumerate(meta["outputs"])}
    return feeds,outputs

def error(ref,actual):
    if ref.shape!=actual.shape or not np.isfinite(actual).all():
        return {"valid":False,"reference_shape":list(ref.shape),"actual_shape":list(actual.shape)}
    if np.issubdtype(ref.dtype,np.integer):
        return {"valid":bool(np.array_equal(ref,actual)),"integer_exact":bool(np.array_equal(ref,actual))}
    a,b=ref.astype(np.float64),actual.astype(np.float64)
    nrms=float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-12))
    result={"valid":nrms<=.01,"nrms":nrms,"max_abs":float(np.max(np.abs(a-b))) if a.size else 0.0}
    return result

def run_stage(directory,captures,out,stage):
    import onnxruntime as ort
    import MNN
    import MNN.nn as nn
    import MNN.expr as expr
    d=Path(directory);root=Path(out);root.mkdir(parents=True,exist_ok=True)
    graph=d/STAGES[stage]
    cases=sorted(Path(captures).glob(stage+"-*.npz"))
    if not cases:raise ValueError("no real captured stage inputs")
    report={"stage":stage,"source_graph":graph.name,"status":"FAIL","cases":[]}
    marker=root/(stage+".json")
    def save():marker.write_text(json.dumps(report,indent=2)+"\n")
    save()
    target=root/(stage+".mnn")
    try:
        # Preserve original numeric formats and reject unsupported conversion.
        # Do not use --allowCustomOp or replace unsupported nodes with stubs.
        args=["mnnconvert","-f","ONNX","--modelFile",str(graph),"--MNNModel",str(target),
              "--keepInputFormat=1"]
        start=time.monotonic()
        with (root/(stage+"-convert.log")).open("w") as log:
            p=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,timeout=600)
        report["conversion_seconds"]=time.monotonic()-start
        report["conversion_exit"]=p.returncode
        if p.returncode or not target.is_file() or target.stat().st_size==0:raise RuntimeError("MNN conversion failed")
        report["converted_bytes"]=target.stat().st_size
        save()
        opt=ort.SessionOptions();opt.intra_op_num_threads=4;opt.inter_op_num_threads=1
        start=time.monotonic()
        baseline=ort.InferenceSession(str(graph),opt,providers=["CPUExecutionProvider"])
        report["ort_load_seconds"]=time.monotonic()-start
        # Time ORT first in isolation, then close it before loading MNN, avoiding
        # two full copies of the large decoder. Timing order is not balanced.
        for path in cases:
            feeds,expected=load_case(path)
            names=list(expected)
            t=[]
            for i in range(4):
                start=time.monotonic();actual=baseline.run(names,feeds);elapsed=time.monotonic()-start
                if i:t.append(elapsed)
            for name,a in zip(names,actual):
                if not error(expected[name],a)["valid"]:raise RuntimeError("captured ORT reference mismatch: "+name)
            report["cases"].append({"case":path.name,"ort_seconds":statistics.median(t),"ort_samples":t,
                                    "input_shapes":{k:list(v.shape) for k,v in feeds.items()}})
        del baseline
        import gc
        gc.collect()
        save()
        feeds,expected=load_case(cases[0])
        inputs=list(feeds);outputs=list(expected)
        start=time.monotonic()
        rt=nn.create_runtime_manager(({"backend":0,"numThread":4,"precision":"high"},))
        module=nn.load_module_from_file(str(target),inputs,outputs,runtime_manager=rt)
        report["mnn_load_seconds"]=time.monotonic()-start
        def variable(a):
            # MNN imports floating FP16 ONNX I/O as FP32; compare all original
            # values after conversion. Exact integer IDs must remain intact.
            if a.dtype==np.float16 or a.dtype==np.float32:
                a=np.ascontiguousarray(a,np.float32);dtype=expr.float
            elif a.dtype==np.int64:
                a=np.ascontiguousarray(a);dtype=expr.int64
            elif a.dtype==np.int32:
                a=np.ascontiguousarray(a);dtype=expr.int
            else:raise ValueError("unsupported input dtype "+str(a.dtype))
            return expr.const(a,list(a.shape),expr.NCHW,dtype)
        for path,row in zip(cases,report["cases"]):
            feeds,expected=load_case(path)
            if set(feeds)!=set(inputs) or list(expected)!=outputs:raise ValueError("dynamic stage interface changed")
            def infer():
                result=module.onForward([variable(feeds[n]) for n in inputs])
                if len(result)!=len(outputs):raise ValueError("MNN output count mismatch")
                return [np.array(v.read(),copy=True) for v in result]
            # Track first call after each shape change separately.
            start=time.monotonic();actual=infer();row["first_shape_seconds"]=time.monotonic()-start
            t=[]
            for _ in range(3):
                start=time.monotonic();actual=infer();t.append(time.monotonic()-start)
            metrics={n:error(expected[n],a) for n,a in zip(outputs,actual)}
            row.update(mnn_seconds=statistics.median(t),mnn_samples=t,outputs=metrics,
                       parity=all(v["valid"] for v in metrics.values()))
            row["speedup"]=row["ort_seconds"]/row["mnn_seconds"]
            if "logits" in expected:
                row["top_id_match"]=bool(np.array_equal(expected["logits"].argmax(-1),actual[outputs.index("logits")].argmax(-1)))
                row["parity"] &= row["top_id_match"]
            if "p_choose" in expected:
                row["read_write_match"]=bool((expected["p_choose"][...,-1,-1].min()>=.5)==(actual[outputs.index("p_choose")][...,-1,-1].min()>=.5))
                row["parity"] &= row["read_write_match"]
            print(json.dumps({"stage":stage,"case":row["case"],"parity":row["parity"],"speedup":row["speedup"],"ort_s":row["ort_seconds"],"mnn_s":row["mnn_seconds"]}),flush=True)
            save()
        report["status"]="PASS" if all(r.get("parity") for r in report["cases"]) else "PARITY_FAIL"
        report["max_rss_mib"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    except Exception as e:
        report["error"]=str(e);report["traceback"]=traceback.format_exc()
    finally:
        save()
        # Reports only; do not publish or retain converted multi-GB models.
        for p in root.glob(target.name+"*"):
            if p.is_file():p.unlink()
    if report["status"]!="PASS":raise RuntimeError(report.get("error","stage parity failed"))

def screen(directory,captures,out):
    root=Path(out);root.mkdir(parents=True,exist_ok=True)
    results=[]
    for stage in STAGES:
        log=root/(stage+"-run.log")
        with log.open("w") as f:
            try:code=subprocess.run([sys.executable,"-u",__file__,"stage",directory,captures,out,stage],
                                    stdout=f,stderr=subprocess.STDOUT,timeout=1800).returncode
            except subprocess.TimeoutExpired:code=124
        marker=root/(stage+".json")
        data=json.loads(marker.read_text()) if marker.exists() else {}
        cases=data.get("cases",[])
        ratio=sum(x.get("mnn_seconds",0) for x in cases)/max(sum(x.get("ort_seconds",0) for x in cases),1e-12) if cases and all("mnn_seconds" in x for x in cases) else None
        row={"stage":stage,"exit":code,"status":data.get("status","FAIL"),"error":data.get("error"),
             "samples":len(cases),"warm_time_ratio":ratio,
             "worth_device_test":bool(code==0 and data.get("status")=="PASS" and ratio is not None and ratio<.8)}
        results.append(row);print(json.dumps(row),flush=True)
        (root/"summary.json").write_text(json.dumps(results,indent=2)+"\n")
    # A screen with negative findings is useful, but never conflate with a
    # passing backend. Require at least one valid whole-stage candidate.
    if not any(r["worth_device_test"] for r in results):raise RuntimeError("no stage cleared parity and 20% speed screen")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("capture","stage","screen"));p.add_argument("directory")
    p.add_argument("input");p.add_argument("output");p.add_argument("stage",nargs="?",choices=STAGES)
    a=p.parse_args()
    if a.mode=="capture":capture(a.directory,a.input,a.output)
    elif a.mode=="stage":run_stage(a.directory,a.input,a.output,a.stage)
    else:screen(a.directory,a.input,a.output)
