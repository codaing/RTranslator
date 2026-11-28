"""Convert real dense bank to float32 and fully integer TFLite.

Evaluate exact fixture on host CPU before making Android benchmark bundle.
Do not treat quantized accuracy or host performance as TPU parity.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf

def metrics(a,b):
    if a.shape!=b.shape or not np.isfinite(b).all():raise ValueError("bad output shape or values")
    d=b.astype(np.float64)-a.astype(np.float64)
    return {"nrms":float(np.linalg.norm(d)/max(np.linalg.norm(a.astype(np.float64)),1e-12)),
            "max_abs":float(np.max(np.abs(d)))}

def convert(source,output):
    source=Path(source);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    reports=[]
    for label in ("single","bank"):
        with np.load(source/(label+".npz")) as f:x,w,y=(f[k] for k in ("x","w","y"))
        meta=json.loads((source/(label+".json")).read_text())
        class Dense(tf.Module):
            @tf.function(input_signature=[tf.TensorSpec(x.shape,tf.float32,name="activation")])
            def __call__(self,a):return tf.linalg.matmul(a,tf.constant(w))
        module=Dense()
        fn=module.__call__.get_concrete_function()
        for precision in ("fp32","int8"):
            name=label+"-"+precision
            converter=tf.lite.TFLiteConverter.from_concrete_functions([fn],module)
            if precision=="int8":
                converter.optimizations=[tf.lite.Optimize.DEFAULT]
                converter.representative_dataset=lambda: iter([[x]])
                converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type=tf.int8;converter.inference_output_type=tf.int8
            blob=converter.convert()
            interp=tf.lite.Interpreter(model_content=blob,num_threads=4,
                experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
            interp.allocate_tensors()
            inp=interp.get_input_details()[0];output=interp.get_output_details()[0]
            ops=[o["op_name"] for o in interp._get_ops_details()]
            if ops!=["FULLY_CONNECTED"]:raise ValueError(f"{name}: expected one fully connected operator, got {ops}")
            if precision=="int8":
                if inp["dtype"]!=np.int8 or output["dtype"]!=np.int8:raise ValueError("not integer I/O")
                scale,zero=inp["quantization"]
                if scale<=0:raise ValueError("invalid input quantization")
                a=np.clip(np.rint(x/scale)+zero,-128,127).astype(np.int8)
            else:a=np.ascontiguousarray(x,dtype="<f4")
            interp.set_tensor(inp["index"],a);interp.invoke()
            raw=interp.get_tensor(output["index"])
            scale,zero=output["quantization"]
            got=(raw.astype(np.float32)-zero)*scale if precision=="int8" else raw
            accuracy=metrics(y,got)
            if precision=="fp32" and accuracy["nrms"]>1e-4:raise ValueError("float conversion parity failed")
            row={**meta,"variant":name,"ops":ops,"model_bytes":len(blob),
                 "input_shape":inp["shape"].tolist(),"input_name":inp["name"],
                 "input_dtype":str(a.dtype),"input_quantization":list(inp["quantization"]),
                 "output_shape":output["shape"].tolist(),"output_dtype":str(raw.dtype),
                 "output_quantization":list(output["quantization"]),
                 "cpu_vs_fp32":accuracy,"precision_screen_pass":accuracy["nrms"]<(.05 if precision=="int8" else 1e-4)}
            (out/(name+".tflite")).write_bytes(blob)
            (out/(name+".input.bin")).write_bytes(a.tobytes())
            (out/(name+".cpu-output.bin")).write_bytes(raw.tobytes())
            (out/(name+".fp32-reference.bin")).write_bytes(y.astype("<f4").tobytes())
            reports.append(row);print(json.dumps(row),flush=True)
    (out/"manifest.json").write_text(json.dumps(reports,indent=2)+"\n")
    # Generate exact flags for the user's already working benchmark APK.
    commands=["# Use one line at a time; allow each run to finish before starting the next.",
              "# Full delegation must be confirmed in logs; never infer from a successful Invoke.",
              "adb shell mkdir -p /data/local/tmp/seamless-tpu",
              "# Run from extracted artifact directory:",
              "adb push . /data/local/tmp/seamless-tpu/"]
    for row in reports:
        name=row["variant"];shape=",".join(map(str,row["input_shape"]))
        # The benchmark permits an arbitrary input label if input order matches.
        common=f"--graph=/data/local/tmp/seamless-tpu/{name}.tflite --input_layer=activation --input_layer_shape={shape} --input_layer_value_files=activation:/data/local/tmp/seamless-tpu/{name}.input.bin --num_threads=4 --warmup_runs=10 --num_runs=100"
        for backend,flags in (("CPU","--use_nnapi=false --use_xnnpack=true"),("TPU","--use_nnapi=true --nnapi_accelerator_name=google-edgetpu --disable_nnapi_cpu=true --use_xnnpack=false --require_full_delegation=true")):
            commands.extend([f"# {name} / {backend}","adb logcat -c",f"adb shell am start -S -n org.tensorflow.lite.benchmark/.BenchmarkModelActivity --es args '\"{common} {flags}\"'","# After completion:", "adb logcat -d > "+name+"-"+backend.lower()+".log"])
    (out/"BENCHMARK-COMMANDS.txt").write_text("\n".join(commands)+"\n")
    (out/"README.txt").write_text("Real SeamlessStreaming projection bank. CC BY-NC 4.0. Static shapes only; not a complete stage. Host reference CPU output provided. Require full delegation and google-edgetpu compile logs. If benchmark version rejects require_full_delegation, do not call it a TPU pass; check full replacement and accelerator compilation explicitly. INT8 accuracy is a screen on calibration fixture only. Do not deploy. CPU and TPU timing must use the same model and input.\n")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("source");p.add_argument("output");a=p.parse_args();convert(a.source,a.output)
