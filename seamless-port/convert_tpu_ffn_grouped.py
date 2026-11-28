"""Per-output-channel-group INT8 decoder FFN experiment.

Split only the final 1024-channel projection into independent output tensors.
Each output group receives its own activation quantization scale, avoiding a
single outlier channel setting the scale for all 1024 outputs. The first
projection remains shared and the 32 outputs stay inside one TFLite graph.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from scipy.optimize import linear_sum_assignment
from convert_tpu_bank import metrics

GROUP=32

def convert(source,output):
    src=Path(source);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((src/"plan.json").read_text())
    with np.load(src/"weights.npz",allow_pickle=False) as f:
        weights={n:f[k] for n,k in meta["constants"].items()}
    cases=[]
    for row in meta["cases"]:
        with np.load(src/row["file"],allow_pickle=False) as f:
            cases.append((f["x"].astype(np.float32),f["y"].astype(np.float32)))
    reports=[]
    for m in (1,32):
        group=[(a,b) for a,b in cases if len(a)==m]
        if len(group)<2:raise ValueError(f"not enough real calls for M={m}")
        x,y=group[-1]
        cal=[a for a,b in group[:-1]]
        if len(cal)>32:cal=[cal[i] for i in np.linspace(0,len(cal)-1,32,dtype=int)]
        shape=(1,1,m,1024)
        def reshape(a):return np.ascontiguousarray(a.reshape(shape),dtype=np.float32)
        class GroupedFFN(tf.Module):
            @tf.function(input_signature=[tf.TensorSpec(shape,tf.float32,name="activation")])
            def __call__(self,a):
                vals={n:tf.constant(v) for n,v in weights.items()}
                vals[meta["input"]]=a
                matmul_index=0
                for step in meta["steps"]:
                    op=step["op"];args=[vals[n] for n in step["inputs"]]
                    if op=="MatMul":
                        v,w=args
                        if matmul_index==0:
                            r=tf.nn.conv2d(v,tf.reshape(w,[1,1,1024,8192]),1,"VALID")
                        elif matmul_index==1:
                            parts=[]
                            for start in range(0,1024,GROUP):
                                filt=tf.reshape(w[:,start:start+GROUP],[1,1,8192,GROUP])
                                parts.append(tf.nn.conv2d(v,filt,1,"VALID"))
                            r=parts
                        else:raise ValueError("unexpected extra FFN MatMul")
                        matmul_index+=1
                    elif op=="Add":
                        left,right=args
                        if isinstance(left,list) or isinstance(right,list):
                            parts=left if isinstance(left,list) else right
                            bias=right if isinstance(left,list) else left
                            if len(parts)!=1024//GROUP:raise ValueError("wrong grouped output count")
                            r=[parts[i]+bias[i*GROUP:(i+1)*GROUP]
                               for i in range(len(parts))]
                        else:r=tf.add(left,right)
                    elif op=="Relu":
                        v=args[0];r=[tf.nn.relu(z) for z in v] if isinstance(v,list) else tf.nn.relu(v)
                    elif op in ("Identity","Cast"):
                        r=args[0]
                    else:raise ValueError("unsupported FFN op for TPU group model: "+op)
                    vals[step["output"]]=r
                result=vals[meta["output"]]
                if not isinstance(result,list) or len(result)!=1024//GROUP:
                    raise ValueError("FFN did not produce expected independent output groups")
                return tuple(result)
        mod=GroupedFFN();fn=mod.__call__.get_concrete_function()
        name=f"ffn-m{m}-conv-g{GROUP}-int8"
        row={"variant":name,"rows":m,"group_size":GROUP,"output_groups":1024//GROUP,
             "calibration_calls":len(cal),"status":"FAIL"}
        try:
            fp=np.concatenate([v.numpy().reshape(m,GROUP) for v in mod(reshape(x))],axis=1)
            row["fp32_vs_source_fp16"]=metrics(y,fp)
            if row["fp32_vs_source_fp16"]["nrms"]>.02:
                raise ValueError("FP32 grouped lowering differs from original FP16 core")
            c=tf.lite.TFLiteConverter.from_concrete_functions([fn],mod)
            c.optimizations=[tf.lite.Optimize.DEFAULT]
            c.representative_dataset=lambda:iter([[reshape(a)] for a in cal])
            c.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            c.inference_input_type=tf.int8;c.inference_output_type=tf.int8
            blob=c.convert()
            interp=tf.lite.Interpreter(model_content=blob,num_threads=4,
                experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
            interp.allocate_tensors()
            inp=interp.get_input_details()[0];outs=interp.get_output_details()
            ops=interp._get_ops_details()
            if sum(op["op_name"]=="CONV_2D" for op in ops)!=1+1024//GROUP:
                raise ValueError("grouped FFN did not lower to one expansion and 32 contraction convolutions")
            if len(outs)!=1024//GROUP:raise ValueError("expected independent output tensors")
            details={t["index"]:t for t in interp.get_tensor_details()}
            scales=[]
            for op in ops:
                if op["op_name"]!="CONV_2D":raise ValueError("unsupported non-convolution op "+op["op_name"])
                w=details[int(op["inputs"][1])];q=w["quantization_parameters"]
                if len(q["scales"])!=int(w["shape"][0]):
                    raise ValueError("convolution filter is not per-output-channel quantized")
                if np.any(q["zero_points"]!=0):raise ValueError("weights are not symmetric")
                scales.append(int(len(q["scales"])))
                for idx in (*op["inputs"],*op["outputs"]):
                    if idx>=0 and details[int(idx)]["dtype"] not in (np.int8,np.int32,np.int64):
                        raise ValueError("non-integer tensor in FFN graph")
            if inp["dtype"]!=np.int8 or any(o["dtype"]!=np.int8 for o in outs):
                raise ValueError("group model is not INT8 at every external tensor")
            s,z=inp["quantization"]
            if s<=0:raise ValueError("invalid input quantization")
            data=np.clip(np.rint(reshape(x)/s)+z,-128,127).astype(np.int8)
            interp.set_tensor(inp["index"],data);interp.invoke()
            parts=[];out_q=[];raw_parts=[];out_names=[]
            for d in outs:
                raw=interp.get_tensor(d["index"]);so,zo=d["quantization"]
                parts.append(((raw.astype(np.float32)-zo)*so).reshape(m,GROUP))
                raw_parts.append(raw.tobytes())
                out_q.append([float(so),int(zo)])
                out_names.append(d["name"])
            # TFLite may reorder multi-output tensors relative to the Python
            # tuple returned by the traced function. Determine the static
            # tensor-to-channel-group mapping against the captured FP16 output.
            cost=np.empty((len(parts),1024//GROUP),dtype=np.float64)
            for i,part in enumerate(parts):
                for j in range(1024//GROUP):
                    ref=y[:,j*GROUP:(j+1)*GROUP]
                    cost[i,j]=np.linalg.norm(part-ref)/max(np.linalg.norm(ref),1e-12)
            tensor_idx,group_idx=linear_sum_assignment(cost)
            group_to_tensor=[-1]*(1024//GROUP)
            for i,j in zip(tensor_idx,group_idx):group_to_tensor[int(j)]=int(i)
            if any(i<0 for i in group_to_tensor) or len(set(group_to_tensor))!=len(parts):
                raise ValueError("could not establish unique TFLite output ordering")
            actual=np.concatenate([parts[i] for i in group_to_tensor],axis=1)
            group_nrms=[float(cost[group_to_tensor[j],j]) for j in range(1024//GROUP)]
            err=metrics(y,actual)
            row.update(status="CONVERTED",model_bytes=len(blob),input_shape=inp["shape"].tolist(),
                output_shapes=[o["shape"].tolist() for o in outs],
                input_name=inp["name"],input_quantization=list(inp["quantization"]),
                output_names=out_names,output_quantizations=out_q,
                output_group_to_tensor_index=group_to_tensor,output_group_nrms=group_nrms,
                filter_scale_counts=scales,ops=[op["op_name"] for op in ops],
                nrms=err["nrms"],max_abs=err["max_abs"],precision_screen_pass=err["nrms"]<.05)
            (out/(name+".tflite")).write_bytes(blob)
            (out/(name+".input.bin")).write_bytes(data.tobytes())
            (out/(name+".cpu-output.bin")).write_bytes(b"".join(raw_parts))
            (out/(name+".source-reference.bin")).write_bytes(y.astype("<f4").tobytes())
        except Exception as exc:
            row["status"]="FAIL";row["error"]=str(exc)
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
            output_path=f"/data/local/tmp/seamless-tpu/{name}-{backend}-output.bin"
            commands.extend(["",f"# {name} / {backend}","adb logcat -c",
                f"""adb shell am start -S -n org.tensorflow.lite.benchmark/.BenchmarkModelActivity --es args '"{common} {flags} --output_filepath={output_path}"'""",
                "# Wait for final Inference timings and output file creation.",
                "sleep 30",f"adb logcat -d -s tflite ExecutionPlan Darwinn > {name}-{backend}.log",
                f"adb pull {output_path} {name}-{backend}-output.bin"])
    (out/"BENCHMARK-COMMANDS.txt").write_text("\n".join(commands)+"\n")
    (out/"README.txt").write_text("Two real dependent FFN projections. Final output projection is split into 32 independently quantized 32-channel outputs to reduce per-tensor output quantization error. One shared expansion convolution; 32 contraction convolutions. Reconstruct FP32 output by dequantizing each output tensor and concatenating groups in order. Same-recording calibration/evaluation only; not held-out speech or full-layer quality. Require full TPU delegation and compare actual output tensors on device. Do not deploy. CC BY-NC 4.0.\n")
    if not all(r["status"]=="CONVERTED" for r in reports):raise RuntimeError("one or more grouped FFN models failed conversion")
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("source");p.add_argument("output")
    a=p.parse_args();convert(a.source,a.output)
