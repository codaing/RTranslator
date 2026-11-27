"""Independent acoustic-unit decoder dynamic INT8 experiment; never stages a pack."""
import argparse
import json
import math
from pathlib import Path
import resource
import time
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic,QuantType
from fold_weights import prepare
from quantize_int8 import phase,rss_mib

def make_fixtures(directory):
    # Use the same pinned checkpoint and frontend, but independently generated
    # examples rather than accepting the quantized graph's own outputs.
    import torch
    from fairseq2.assets import asset_store
    from seamless_communication.models.unity import load_unity_config,load_unity_model,load_unity_text_tokenizer
    name="seamless_streaming_unity"
    cfg=load_unity_config(name)
    cfg.use_text_decoder=False;cfg.use_text_encoder=False
    card=asset_store.retrieve_card(name);card.field("model_config").set(cfg)
    model=load_unity_model(card,device=torch.device("cpu"),dtype=torch.float32).eval().t2u_model
    encoder=load_unity_text_tokenizer(name).create_encoder(lang="eng",mode="target")
    torch.manual_seed(12)
    with torch.inference_mode():
        for i,text in enumerate(("a number","the square root of two","the derivative of a function is zero",
                "the limit of the sequence is infinity","the integral from zero to one of x squared")):
            tokens=encoder(text).reshape(1,-1)
            chars,_,lens=model.decoder_frontend.text_to_char_seqs(tokens.clone())
            features=torch.randn(1,tokens.shape[1],model.model_dim)
            output,_,durations=model(features,None,tokens.clone(),duration_factor=1.0)
            np.savez(directory/f"units.extra{i}.npz",features=features.numpy(),chars=chars.numpy(),
                     char_lens=lens.numpy(),logits=output.logits.numpy(),durations=durations.numpy())

def graph_bytes(path):
    g=onnx.load(str(path),load_external_data=False)
    files={path}
    from weight_only_int8 import tensors
    for t in tensors(g.graph):
        for e in t.external_data:
            if e.key=="location":files.add(path.parent/e.value)
    return sum(f.stat().st_size for f in files)

def evaluate(directory):
    source=directory/"units.onnx"; output=directory/"units.int8.onnx"
    report=directory/"units-int8-experiment.json"
    baseline=json.loads((directory/"units.validated.json").read_text())
    if baseline.get("parity")!="PASS" or baseline.get("precision")!="fp32":
        raise ValueError("validated units FP32 required")
    fixtures=sorted(directory.glob("units.extra*.npz"))
    if len(fixtures)!=5:raise ValueError("expected five reference fixtures")
    opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1
    results={"component":"units","precision":"int8-dynamic","parity":"PENDING","fp32_bytes":graph_bytes(source),"tests":[]}
    def save():report.write_text(json.dumps(results,indent=2)+"\n")
    save()
    # Check that independent reference inputs are valid for the FP32 ONNX
    # graph before attributing any numerical error to quantization.
    with phase("independent FP32 fixture comparison"):
        session=ort.InferenceSession(str(source),sess_options=opts,providers=["CPUExecutionProvider"])
        for path in fixtures:
            with np.load(path,allow_pickle=False) as f:
                logits,durations=session.run(["logits","durations"],{k:f[k] for k in ("features","chars","char_lens")})
                np.testing.assert_array_equal(durations,f["durations"])
                np.testing.assert_allclose(logits,f["logits"],rtol=0.02,atol=0.002)
                np.testing.assert_array_equal(logits.argmax(-1),f["logits"].argmax(-1))
        del session
    with phase("units INT8 quantization"):
        folded=prepare(source)
        quantize_dynamic(str(folded),str(output),weight_type=QuantType.QInt8,per_channel=True,
                         op_types_to_quantize=["MatMul","Gemm"],extra_options={"MatMulConstBOnly":True},
                         use_external_data_format=True)
    onnx.checker.check_model(str(output))
    g=onnx.load(str(output),load_external_data=False)
    results["quantized_ops"]=sum(n.op_type in ("MatMulInteger","QLinearMatMul","DynamicQuantizeMatMul","QGemm") for n in g.graph.node)
    if not results["quantized_ops"]:raise RuntimeError("no integer weight operations")
    results["int8_bytes"]=graph_bytes(output)
    save();del g
    with phase("INT8 session load"):
        session=ort.InferenceSession(str(output),sess_options=opts,providers=["CPUExecutionProvider"])
    results["session_rss_mib"]=rss_mib()
    failures=[]
    for path in fixtures:
        with phase(path.name),np.load(path,allow_pickle=False) as f:
            inputs={k:f[k] for k in ("features","chars","char_lens")}
            session.run(None,inputs)
            start=time.monotonic()
            logits,durations=session.run(["logits","durations"],inputs)
            elapsed=time.monotonic()-start
            exact_duration=bool(np.array_equal(durations,f["durations"]))
            metrics={"fixture":path.name,"duration_exact":exact_duration,"seconds":elapsed,"rss_mib":rss_mib(),
                     "peak_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                     "reference_shape":list(f["logits"].shape),"actual_shape":list(logits.shape)}
            if not exact_duration:failures.append(path.name+": duration mismatch")
            if logits.shape!=f["logits"].shape or not np.isfinite(logits).all():
                failures.append(path.name+": invalid logits shape or values")
            else:
                expected=f["logits"].astype(np.float64)
                nrms=float(np.sqrt(np.mean((logits-expected)**2)))/max(float(np.sqrt(np.mean(expected**2))),1e-6)
                agreement=float(np.mean(logits.argmax(-1)==expected.argmax(-1)))
                metrics.update(logits_nrms=nrms,unit_agreement=agreement)
                if nrms>0.10 or agreement!=1.0:failures.append(path.name+": unit/logit parity")
            results["tests"].append(metrics);results["failures"]=failures
            print(json.dumps(metrics),flush=True);save()
    results["parity"]="FAIL" if failures else "PASS";save()
    if failures:raise RuntimeError("; ".join(failures))

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("fixtures","evaluate"));p.add_argument("directory",type=Path)
    a=p.parse_args()
    (make_fixtures if a.mode=="fixtures" else evaluate)(a.directory)
