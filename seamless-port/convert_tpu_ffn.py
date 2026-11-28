"""Compare fused real FFN core as per-channel Conv2D or per-tensor Dense.

Two dependent matrix operations plus actual activation and biases; quantize
intermediate activations inside the graph. Use real decoder calls for calibration
and disjoint calls of the same recording for evaluation. Not held-out speech.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from convert_tpu_bank import metrics

def convert(source,output):
    src=Path(source);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((src/"plan.json").read_text())
    with np.load(src/"weights.npz",allow_pickle=False) as f:
        weights={n:f[k] for n,k in meta["constants"].items()}
    cases=[]
    for row in meta["cases"]:
        with np.load(src/row["file"],allow_pickle=False) as f:cases.append((f["x"],f["y"]))
    counts=sorted(set(len(a) for a,b in cases))
    # Pick autoregressive M=1 and the most frequent nontrivial prefill size
    # with at least two real calls. No synthetic padding of model context.
    sizes=[m for m in counts if sum(len(a)==m for a,b in cases)>=2]
    if not sizes:raise ValueError("no repeated real FFN shapes")
    prefill=[m for m in sizes if m>1]
    sizes=([1] if 1 in sizes else [])+([max(prefill,key=lambda m:(sum(len(a)==m for a,b in cases),m))] if prefill else [])
    reports=[]
    for m in sizes:
        group=[(a,b) for a,b in cases if len(a)==m]
        # Separate eval call from all calibration calls of this shape.
        x,y=group[-1];cal=[a for a,b in group[:-1]]
        if len(cal)>32:cal=[cal[i] for i in np.linspace(0,len(cal)-1,32,dtype=int)]
        for layout in ("conv","dense"):
            shape=(1,1,m,1024) if layout=="conv" else (m,1024)
            def reshape(a):return np.ascontiguousarray(a.reshape(shape),dtype=np.float32)
            class FFN(tf.Module):
                @tf.function(input_signature=[tf.TensorSpec(shape,tf.float32,name="activation")])
                def __call__(self,a):
                    vals={n:tf.constant(v) for n,v in weights.items()}
                    vals[meta["input"]]=a
                    for step in meta["steps"]:
                        op=step["op"];args=[vals[n] for n in step["inputs"]]
                        if op=="MatMul":
                            v,w=args
                            if layout=="conv":r=tf.nn.conv2d(v,tf.reshape(w,[1,1,w.shape[0],w.shape[1]]),1,"VALID")
                            else:r=tf.linalg.matmul(v,w)
                        elif op=="Gelu":r=tf.nn.gelu(args[0],approximate=step["attrs"]["approximate"]=="tanh")
                        elif op in ("Identity","Cast"):r=args[0]
                        else:
                            fn={"Add":tf.add,"Relu":tf.nn.relu,"Erf":tf.math.erf,"Mul":tf.multiply,
                                "Div":tf.divide,"Sub":tf.subtract,"Pow":tf.pow,"Tanh":tf.tanh,
                                "Sigmoid":tf.sigmoid,"Neg":tf.negative,"Sqrt":tf.sqrt}[op]
                            r=fn(*args)
                        vals[step["output"]]=r
                    return vals[meta["output"]]
            mod=FFN();fn=mod.__call__.get_concrete_function()
            name=f"ffn-m{m}-{layout}-int8"
            row={"variant":name,"rows":m,"calibration_calls":len(cal),"status":"FAIL"}
            try:
                fp=mod(reshape(x)).numpy().reshape(y.shape)
                row["fp32_vs_source_fp16"]=metrics(y,fp)
                if row["fp32_vs_source_fp16"]["nrms"]>.02:raise ValueError("FP32 lowering does not match original FP16 core")
                c=tf.lite.TFLiteConverter.from_concrete_functions([fn],mod)
                c.optimizations=[tf.lite.Optimize.DEFAULT];c.representative_dataset=lambda:iter([[reshape(a)] for a in cal])
                c.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                c.inference_input_type=tf.int8;c.inference_output_type=tf.int8
                if layout=="dense":c._experimental_disable_per_channel=True
                blob=c.convert()
                interp=tf.lite.Interpreter(model_content=blob,num_threads=4,
                    experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
                interp.allocate_tensors()
                inp=interp.get_input_details()[0];o=interp.get_output_details()[0]
                ops=interp._get_ops_details();expected="CONV_2D" if layout=="conv" else "FULLY_CONNECTED"
                if sum(op["op_name"]==expected for op in ops)!=2:raise ValueError("two dependent projections did not survive lowering")
                details={t["index"]:t for t in interp.get_tensor_details()}
                for op in ops:
                    for i in (*op["inputs"],*op["outputs"]):
                        if i>=0 and details[int(i)]["dtype"] not in (np.int8,np.int32,np.int64):
                            raise ValueError("non-integer operator tensor")
                    if op["op_name"]==expected:
                        w=details[int(op["inputs"][1])];q=w["quantization_parameters"]
                        target=w["shape"][0] if layout=="conv" else 1
                        if len(q["scales"])!=target:raise ValueError("unexpected weight quantization layout")
                if inp["dtype"]!=np.int8 or o["dtype"]!=np.int8:raise ValueError("not INT8 IO")
                s,z=inp["quantization"]
                if s<=0:raise ValueError("invalid scale")
                data=np.clip(np.rint(reshape(x)/s)+z,-128,127).astype(np.int8)
                interp.set_tensor(inp["index"],data);interp.invoke()
                raw=interp.get_tensor(o["index"]);s,z=o["quantization"]
                actual=((raw.astype(np.float32)-z)*s).reshape(y.shape)
                err=metrics(y,actual)
                row.update(status="CONVERTED",model_bytes=len(blob),input_shape=inp["shape"].tolist(),
                    output_shape=o["shape"].tolist(),input_name=inp["name"],
                    input_quantization=list(inp["quantization"]),output_quantization=list(o["quantization"]),
                    ops=[op["op_name"] for op in ops],nrms=err["nrms"],max_abs=err["max_abs"],
                    precision_screen_pass=err["nrms"]<.05)
                (out/(name+".tflite")).write_bytes(blob)
                (out/(name+".input.bin")).write_bytes(data.tobytes())
                (out/(name+".cpu-output.bin")).write_bytes(raw.tobytes())
                (out/(name+".source-reference.bin")).write_bytes(y.astype("<f4").tobytes())
            except Exception as exc:row["error"]=str(exc)
            reports.append(row);print(json.dumps(row),flush=True)
            (out/"manifest.json").write_text(json.dumps(reports,indent=2)+"\n")
    commands=["# From extracted artifact directory","adb shell mkdir -p /data/local/tmp/seamless-tpu","adb push . /data/local/tmp/seamless-tpu/"]
    for row in reports:
        if row["status"]!="CONVERTED":continue
        name=row["variant"];shape=",".join(map(str,row["input_shape"]))
        common=(f"--graph=/data/local/tmp/seamless-tpu/{name}.tflite --input_layer=activation "
                f"--input_layer_shape={shape} --input_layer_value_files=activation:/data/local/tmp/seamless-tpu/{name}.input.bin "
                "--num_threads=4 --warmup_runs=10 --num_runs=100")
        for backend,flags in (("tpu","--use_nnapi=true --nnapi_accelerator_name=google-edgetpu --disable_nnapi_cpu=true --use_xnnpack=false --require_full_delegation=true"),
                              ("cpu","--use_nnapi=false --use_xnnpack=true")):
            commands.extend(["",f"# {name} / {backend}","adb logcat -c",
                f"""adb shell am start -S -n org.tensorflow.lite.benchmark/.BenchmarkModelActivity --es args '"{common} {flags}"'""",
                "# Wait for final Inference timings or an explicit delegate failure (not just warmup).",
                "sleep 30",f"adb logcat -d -s tflite ExecutionPlan Darwinn > {name}-{backend}.log"])
    (out/"BENCHMARK-COMMANDS.txt").write_text("\n".join(commands)+"\n")
    (out/"README.txt").write_text("Real two-projection decoder FFN core, not a complete transformer layer. "
        "Calibration and evaluation use distinct calls of one speech fixture, not held-out speech. "
        "Host numerical comparison and quantization inventory only. Require full TPU delegation and device output parity. Do not deploy. CC BY-NC 4.0.\n")
    if not any(r["status"]=="CONVERTED" for r in reports):raise RuntimeError("no FFN model converted")
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("source");p.add_argument("output")
    a=p.parse_args();convert(a.source,a.output)
