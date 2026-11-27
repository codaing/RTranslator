#!/usr/bin/env python3
"""Selective INT8 quantization with unchanged accuracy gates and progress reporting."""
import argparse
import contextlib
import json
import math
from pathlib import Path
import resource
import threading
import time


def rss_mib():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])/1024
    except OSError:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024


@contextlib.contextmanager
def phase(label):
    start=time.monotonic()
    stop=threading.Event()
    print(f"START {label} RSS={rss_mib():.0f} MiB",flush=True)
    def heartbeat():
        while not stop.wait(30):
            print(f"RUNNING {label} elapsed={time.monotonic()-start:.0f}s RSS={rss_mib():.0f} MiB",flush=True)
    worker=threading.Thread(target=heartbeat,daemon=True)
    worker.start()
    try:
        yield
    finally:
        stop.set()
        worker.join()
        print(f"END {label} elapsed={time.monotonic()-start:.1f}s peak_RSS={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f} MiB",flush=True)


def select_nodes(graph,component):
    """Never quantize attention/monotonic/output projections in the decoder."""
    import re
    import onnx
    initializers={t.name:t for t in graph.initializer}
    entries=[]
    for n in graph.node:
        if n.op_type not in ("MatMul","Gemm") or len(n.input)<2 or n.input[1] not in initializers:
            continue
        weight=initializers[n.input[1]]
        if weight.data_type!=onnx.TensorProto.FLOAT or len(weight.dims)!=2:
            continue
        # fairseq2 StandardFeedForwardNetwork has ffn.inner_proj and
        # ffn.output_proj. Match full module components, not a vague substring.
        selected=component=="encoder" or bool(re.search(r"(?:^|/)ffn/(?:inner_proj|output_proj)(?:/|$)",n.name))
        entries.append({"name":n.name,"weight":weight.name,"shape":list(weight.dims),
                        "elements":math.prod(weight.dims),"selected":selected})
    return entries


def quantize(component, directory, native=False, asymmetric=False, subset="all"):
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    directory = Path(directory)
    source = directory / f"{component}.onnx"
    report = directory / f"{component}.validated.json"
    if native and component!="decoder":raise ValueError("native experiment is decoder only")
    suffix=("native8-"+("asymmetric" if asymmetric else "symmetric")+"-"+subset) if native else "int8"
    output=directory/f"{component}.{suffix}.onnx"
    marker=directory/f"{component}.{suffix}.validated.json"
    diagnostics=directory/f"{component}.{suffix}.diagnostics.json"
    marker.unlink(missing_ok=True)
    diagnostics.unlink(missing_ok=True)
    baseline = json.loads(report.read_text())
    if baseline.get("parity") != "PASS" or baseline.get("precision") != "fp32" or baseline.get("component") != component:
        raise ValueError("A validated FP32 graph is required")
    cases = sorted(directory.glob(f"{component}.case*.npz"))
    if not cases or len(cases)!=len(baseline.get("tests",[])):
        raise ValueError("Missing or mismatched FP32 validation fixtures")
    from fold_weights import prepare
    with phase("constant weight preparation"):
        source = prepare(source)
    with phase("quantization selection"):
        graph=onnx.load(str(source),load_external_data=False).graph
        entries=select_nodes(graph,component)
        # Weight-only decoder path currently supports MatMul with groups of 16.
        matmul_names={n.name for n in graph.node if n.op_type=="MatMul"}
        if component=="decoder":
            for e in entries:
                e["selected"]=e["selected"] and e["name"] in matmul_names and e["shape"][0]%16==0
        if native and subset!="all":
            for e in entries:
                e["selected"]=e["selected"] and ("/ffn/"+subset+"/") in e["name"]
        selected=[e["name"] for e in entries if e["selected"]]
        if component=="decoder" and (not selected or "" in selected or len(set(selected))!=len(selected)):
            raise RuntimeError("Could not uniquely identify decoder feed-forward projections")
        policy="ffn-group16-weight-only-fp32-activations" if component=="decoder" else "all-constant-matmul-int8"
        if native:policy="native8-group16-"+("asymmetric" if asymmetric else "symmetric")+"-"+subset
        summary={"component":component,"precision":"int8-native-experimental" if native else ("int8-weight-only" if component=="decoder" else "int8-dynamic"),"policy":policy,"parity":"PENDING",
                 "eligible":len(entries),"selected":len(selected),"node_selection":entries,"tests":[]}
        diagnostics.write_text(json.dumps(summary,indent=2)+"\n")
        print(f"Policy={policy} selected={len(selected)}/{len(entries)} matrix ops; "
              f"selected weight elements={sum(e['elements'] for e in entries if e['selected']):,}",flush=True)
        del graph
    if component=="decoder":
        from weight_only_int8 import convert
        with phase("grouped weight-only INT8 conversion"):
            convert(source,output,selected,group=16,native=native,asymmetric=asymmetric)
    else:
        with phase("ORT dynamic weight quantization"):
            quantize_dynamic(str(source),str(output),weight_type=QuantType.QInt8,
                per_channel=True,op_types_to_quantize=["MatMul","Gemm"],
                extra_options={"MatMulConstBOnly":True},use_external_data_format=True)
    with phase("ONNX checker and graph inspection"):
        onnx.checker.check_model(str(output))
        graph=onnx.load(str(output),load_external_data=False)
        integer_ops=("MatMulInteger","QLinearMatMul","DynamicQuantizeMatMul","QGemm","DynamicQuantizeLinear","QuantizeLinear")
        if component=="decoder":
            if any(n.op_type in integer_ops for n in graph.graph.node):
                raise RuntimeError("unexpected activation quantization in weight-only graph")
            quantized_ops=sum(n.op_type=="MatMulNBits" and n.domain=="com.microsoft" for n in graph.graph.node) if native else sum(n.op_type=="DequantizeLinear" and n.name.startswith("weight_only/") for n in graph.graph.node)
            if native and quantized_ops!=len(selected):raise RuntimeError("native operator count mismatch")
        else:
            quantized_ops=sum(n.op_type in integer_ops[:4] for n in graph.graph.node)
        if not quantized_ops:raise RuntimeError("No INT8 weight operations produced")
        summary["quantized_ops"]=quantized_ops
        summary["initializer_bytes_by_dtype"]={}
        for t in graph.graph.initializer:
            key=onnx.TensorProto.DataType.Name(t.data_type)
            size=math.prod(t.dims)*{onnx.TensorProto.FLOAT:4,onnx.TensorProto.FLOAT16:2,onnx.TensorProto.INT8:1,onnx.TensorProto.UINT8:1,onnx.TensorProto.INT64:8,onnx.TensorProto.INT32:4}.get(t.data_type,0)
            summary["initializer_bytes_by_dtype"][key]=summary["initializer_bytes_by_dtype"].get(key,0)+size
        del graph
    options=ort.SessionOptions()
    options.intra_op_num_threads=4
    options.inter_op_num_threads=1
    if native:
        # Keep native operator available, avoid a second retained packed copy
        # where possible, and measure rather than assume memory improvement.
        from decoder_session import configure_native
        configure_native(options)
    elif component=="decoder":
        from decoder_session import configure
        configure(options)
    with phase("CPU session creation"):
        session=ort.InferenceSession(str(output),sess_options=options,providers=["CPUExecutionProvider"])
    summary["session_rss_mib"]=rss_mib()
    summary["peak_conversion_rss_mib"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    diagnostics.write_text(json.dumps(summary,indent=2)+"\n")
    failures=[]
    for case in cases:
        with phase("fixture "+case.name),np.load(case,allow_pickle=False) as fixture:
            inputs={i.name:fixture["input_"+i.name] for i in session.get_inputs()}
            names=[o.name for o in session.get_outputs()]
            session.run(names,inputs)
            start=time.monotonic()
            values=session.run(names,inputs)
            elapsed=time.monotonic()-start
            metrics={}
            for name,actual in zip(names,values):
                expected=fixture["output_"+name]
                if expected.shape!=actual.shape or not np.isfinite(actual).all():
                    failures.append(f"{case.name}: invalid {name} output")
                    metrics[name]={"valid":False}
                    continue
                delta=actual.astype(np.float64)-expected.astype(np.float64)
                nrms=float(np.sqrt(np.mean(delta**2)))/max(float(np.sqrt(np.mean(expected.astype(np.float64)**2))),1e-6)
                metrics[name]={"normalized_rms_error":nrms,"max_absolute_error":float(np.max(np.abs(delta)))}
                if nrms>0.10:failures.append(f"{case.name}: {name} normalized RMS {nrms:.3f} exceeds 0.10")
                if name=="logits":
                    agreement=float(np.mean(np.argmax(actual,axis=-1)==np.argmax(expected,axis=-1)))
                    metrics[name]["top_token_agreement"]=agreement
                    if agreement<1.0:failures.append(f"{case.name}: top token differs")
                if name=="p_choose":
                    decision=bool(np.min(actual[..., -1,-1])>=0.5)
                    reference=bool(np.min(expected[..., -1,-1])>=0.5)
                    metrics[name].update(read_write_agreement=decision==reference,
                        actual_probability=float(np.min(actual[..., -1,-1])),reference_probability=float(np.min(expected[..., -1,-1])))
                    if decision!=reference:failures.append(f"{case.name}: monotonic read/write decision differs")
            summary["tests"].append({"fixture":case.name,"seconds":elapsed,"metrics":metrics,"rss_mib":rss_mib(),"peak_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024})
            summary["failures"]=failures
            print(json.dumps(summary["tests"][-1]),flush=True)
            diagnostics.write_text(json.dumps(summary,indent=2)+"\n")
    summary["parity"]="FAIL" if failures else "PASS"
    summary["final_rss_mib"]=rss_mib()
    summary["peak_inference_and_conversion_rss_mib"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    diagnostics.write_text(json.dumps(summary,indent=2)+"\n")
    if failures:raise RuntimeError("; ".join(failures))
    marker.write_text(json.dumps(summary,indent=2)+"\n")
    return marker


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("component",choices=("encoder","decoder"))
    p.add_argument("directory",type=Path)
    p.add_argument("--native",action="store_true")
    p.add_argument("--asymmetric",action="store_true")
    p.add_argument("--subset",choices=("all","inner_proj","output_proj"),default="all")
    args=p.parse_args()
    print(quantize(args.component,args.directory,native=args.native,asymmetric=args.asymmetric,subset=args.subset))


if __name__=="__main__":
    main()
