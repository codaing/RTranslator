"""Equivalent 1x1 Conv2D lowering of real decoder projections.

NNAPI supports per-channel filters for Conv2D, unlike FullyConnected.
Keep the same real input, weights and FP32 reference as the dense probe.
No claim of device support or speech parity from host conversion.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from convert_tpu_bank import metrics

def convert(source,output):
    source=Path(source);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for label in ("single","bank"):
        with np.load(source/(label+".npz")) as f:
            x,w,y=(f[k] for k in ("x","w","y"))
        meta=json.loads((source/(label+".json")).read_text())
        m,k=x.shape;n=w.shape[1]
        a=np.ascontiguousarray(x.reshape(1,1,m,k),dtype=np.float32)
        filter_=np.ascontiguousarray(w.reshape(1,1,k,n),dtype=np.float32)
        class Pointwise(tf.Module):
            @tf.function(input_signature=[tf.TensorSpec(a.shape,tf.float32,name="activation")])
            def __call__(self,v):
                return tf.nn.conv2d(v,tf.constant(filter_),strides=1,padding="VALID")
        mod=Pointwise()
        for variant in ("fp32","int8-pc"):
            name=label+"-conv-"+variant
            converter=tf.lite.TFLiteConverter.from_concrete_functions([mod.__call__.get_concrete_function()],mod)
            if variant=="int8-pc":
                converter.optimizations=[tf.lite.Optimize.DEFAULT]
                converter.representative_dataset=lambda: iter([[a]])
                converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type=tf.int8;converter.inference_output_type=tf.int8
            blob=converter.convert()
            interp=tf.lite.Interpreter(model_content=blob,num_threads=4,
                experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
            interp.allocate_tensors()
            op=interp._get_ops_details()
            if [v["op_name"] for v in op]!=["CONV_2D"]:
                raise ValueError(f"{name}: unexpected lowering "+str([v["op_name"] for v in op]))
            inp=interp.get_input_details()[0];output_detail=interp.get_output_details()[0]
            wt=next(t for t in interp.get_tensor_details() if t["index"]==int(op[0]["inputs"][1]))
            q=wt["quantization_parameters"];scales=len(q["scales"])
            if variant=="int8-pc":
                if inp["dtype"]!=np.int8 or output_detail["dtype"]!=np.int8 or scales!=n or q["quantized_dimension"]!=0:
                    raise ValueError("expected signed integer I/O and per-output-channel filter scales")
                if np.any(q["zero_points"]!=0) or np.any(q["scales"]<=0):
                    raise ValueError("expected symmetric per-channel weights")
                s,z=inp["quantization"]
                if s<=0:raise ValueError("invalid input scale")
                data=np.clip(np.rint(a/s)+z,-128,127).astype(np.int8)
            else:data=a
            interp.set_tensor(inp["index"],data);interp.invoke()
            raw=interp.get_tensor(output_detail["index"])
            s,z=output_detail["quantization"]
            actual=(raw.astype(np.float32)-z)*s if variant=="int8-pc" else raw
            if actual.shape!=(1,1,m,n):raise ValueError("unexpected output shape")
            err=metrics(y,actual.reshape(m,n))
            if variant=="fp32":
                if err["nrms"]>1e-4:raise ValueError("pointwise lowering changed FP32 output")
                continue
            row={**meta,"variant":name,"ops":["CONV_2D"],"input_shape":list(data.shape),
                 "output_shape":list(raw.shape),"input_name":inp["name"],"input_dtype":str(data.dtype),
                 "input_quantization":list(inp["quantization"]),"output_quantization":list(output_detail["quantization"]),
                 "weight_shape":wt["shape"].tolist(),"weight_scale_count":scales,
                 "model_bytes":len(blob),"cpu_vs_fp32":err,"precision_screen_pass":err["nrms"]<.05}
            (out/(name+".tflite")).write_bytes(blob)
            (out/(name+".input.bin")).write_bytes(data.tobytes())
            (out/(name+".cpu-output.bin")).write_bytes(raw.tobytes())
            (out/(name+".fp32-reference.bin")).write_bytes(y.astype("<f4").tobytes())
            rows.append(row);print(json.dumps(row),flush=True)
    (out/"manifest.json").write_text(json.dumps(rows,indent=2)+"\n")
    commands=["# From extracted artifact directory:", "adb shell mkdir -p /data/local/tmp/seamless-tpu",
              "adb push . /data/local/tmp/seamless-tpu/"]
    for row in reversed(rows):
        name=row["variant"];shape=",".join(map(str,row["input_shape"]))
        common=(f"--graph=/data/local/tmp/seamless-tpu/{name}.tflite --input_layer=activation "
                f"--input_layer_shape={shape} --input_layer_value_files=activation:/data/local/tmp/seamless-tpu/{name}.input.bin "
                "--num_threads=4 --warmup_runs=10 --num_runs=100")
        for backend,flags in (("tpu","--use_nnapi=true --nnapi_accelerator_name=google-edgetpu --disable_nnapi_cpu=true --use_xnnpack=false --require_full_delegation=true"),
                              ("cpu","--use_nnapi=false --use_xnnpack=true")):
            commands.extend(["",f"# {name} {backend}", "adb logcat -c",
                f"""adb shell am start -S -n org.tensorflow.lite.benchmark/.BenchmarkModelActivity --es args '"{common} {flags}"'""",
                "# Wait until the final inference timings are printed or delegation fails, then:",
                f"adb logcat -d > {name}-{backend}.log"])
    (out/"BENCHMARK-COMMANDS.txt").write_text("\n".join(commands)+"\n")
    (out/"README.txt").write_text("Real-weight pointwise Conv2D lowering, same 96 speech frames and weights as dense bank. "
        "FP32 equivalent verified on host; INT8 per-channel weights verified. "
        "No Pixel TPU support, actual output parity, or speed claim yet. "
        "Calibration on same fixture only; do not deploy. CC BY-NC 4.0.\n")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("source");p.add_argument("output")
    a=p.parse_args();convert(a.source,a.output)
