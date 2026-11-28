"""Build real-weight FP16 and NNAPI-compatible INT8 TFLite projection probes."""
import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf

def metrics(a, b):
    if a.shape != b.shape or not np.isfinite(b).all():
        raise ValueError("bad output shape or values")
    d = b.astype(np.float64) - a.astype(np.float64)
    return {"nrms": float(np.linalg.norm(d) / max(np.linalg.norm(a.astype(np.float64)), 1e-12)),
            "max_abs": float(np.max(np.abs(d)))}

def convert(source, output):
    source = Path(source); out = Path(output); out.mkdir(parents=True, exist_ok=True)
    reports = []
    for label in ("single", "bank"):
        with np.load(source / (label + ".npz")) as f:
            x, w, y = (f[k] for k in ("x", "w", "y"))
            source_y = f["source_y"] if "source_y" in f.files else y
        meta = json.loads((source / (label + ".json")).read_text())

        class Dense32(tf.Module):
            @tf.function(input_signature=[tf.TensorSpec(x.shape, tf.float32, name="activation")])
            def __call__(self, a):
                return tf.linalg.matmul(a, tf.constant(w, dtype=tf.float32))

        class Dense16(tf.Module):
            @tf.function(input_signature=[tf.TensorSpec(x.shape, tf.float16, name="activation")])
            def __call__(self, a):
                return tf.linalg.matmul(a, tf.constant(w.astype(np.float16), dtype=tf.float16))

        models = {
            "fp32": (Dense32(), np.ascontiguousarray(x, dtype="<f4"), y, "fp32-reference"),
            "fp16": (Dense16(), np.ascontiguousarray(x, dtype="<f2"), source_y, "source-fp16-reference"),
        }
        for variant in ("fp32", "fp16", "int8-pc", "int8-pt"):
            name = label + "-" + variant
            try:
                if variant in ("fp32", "fp16"):
                    module, input_data, reference, reference_kind = models[variant]
                    converter = tf.lite.TFLiteConverter.from_concrete_functions(
                        [module.__call__.get_concrete_function()], module)
                else:
                    module = models["fp32"][0]
                    input_data = None; reference = y; reference_kind = "fp32-reference"
                    converter = tf.lite.TFLiteConverter.from_concrete_functions(
                        [module.__call__.get_concrete_function()], module)
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                    converter.representative_dataset = lambda: iter([[x]])
                    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                    converter.inference_input_type = tf.int8
                    converter.inference_output_type = tf.int8
                    if variant == "int8-pt":
                        converter._experimental_disable_per_channel = True

                blob = converter.convert()
                interp = tf.lite.Interpreter(
                    model_content=blob, num_threads=4,
                    experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
                interp.allocate_tensors()
                inp = interp.get_input_details()[0]
                output_detail = interp.get_output_details()[0]
                ops = [op["op_name"] for op in interp._get_ops_details()]
                if ops != ["FULLY_CONNECTED"]:
                    raise ValueError(f"expected one FULLY_CONNECTED op, got {ops}")

                if variant.startswith("int8"):
                    if inp["dtype"] != np.int8 or output_detail["dtype"] != np.int8:
                        raise ValueError("not integer I/O")
                    scale, zero = inp["quantization"]
                    if scale <= 0: raise ValueError("invalid input quantization")
                    input_data = np.clip(np.rint(x / scale) + zero, -128, 127).astype(np.int8)

                op_detail = interp._get_ops_details()[0]
                weight_index = int(op_detail["inputs"][1])
                weight_detail = next(t for t in interp.get_tensor_details() if t["index"] == weight_index)
                weight_q = weight_detail["quantization_parameters"]
                weight_scales = np.asarray(weight_q["scales"])
                if variant == "int8-pt" and len(weight_scales) != 1:
                    raise ValueError(f"per-tensor request still has {len(weight_scales)} weight scales")

                interp.set_tensor(inp["index"], input_data); interp.invoke()
                raw = interp.get_tensor(output_detail["index"])
                out_scale, out_zero = output_detail["quantization"]
                got = ((raw.astype(np.float32) - out_zero) * out_scale
                       if variant.startswith("int8") else raw.astype(np.float32))
                accuracy = metrics(reference, got)
                row = {
                    **meta, "variant": name, "ops": ops, "model_bytes": len(blob),
                    "input_shape": inp["shape"].tolist(), "input_name": inp["name"],
                    "input_dtype": str(input_data.dtype), "input_quantization": list(inp["quantization"]),
                    "weight_dtype": str(weight_detail["dtype"]),
                    "weight_quantization_scale_count": int(len(weight_scales)),
                    "weight_quantization_zero_points": np.asarray(weight_q["zero_points"]).tolist(),
                    "output_shape": output_detail["shape"].tolist(), "output_dtype": str(raw.dtype),
                    "output_quantization": list(output_detail["quantization"]),
                    "reference_kind": reference_kind, "cpu_vs_reference": accuracy,
                    "precision_screen_pass": accuracy["nrms"] < (0.05 if variant.startswith("int8") else 1e-4),
                }
                (out / (name + ".tflite")).write_bytes(blob)
                (out / (name + ".input.bin")).write_bytes(input_data.tobytes())
                (out / (name + ".cpu-output.bin")).write_bytes(raw.tobytes())
                (out / (name + ".reference.bin")).write_bytes(reference.astype("<f4").tobytes())
                reports.append(row); print(json.dumps(row), flush=True)
            except Exception as e:
                reports.append({"variant": name, "conversion_error": repr(e)})
                print(json.dumps(reports[-1]), flush=True)

    (out / "manifest.json").write_text(json.dumps(reports, indent=2) + "\n")
    commands = [
        "# Run each candidate separately. Allow the 100-run benchmark to finish.",
        "# TPU pass requires full delegation and successful google-edgetpu compilation.",
        "adb shell mkdir -p /data/local/tmp/seamless-tpu",
        "# Run from extracted artifact directory:",
        "adb push . /data/local/tmp/seamless-tpu/",
    ]
    for label in ("single", "bank"):
        for variant in ("fp16", "int8-pt"):
            name = label + "-" + variant
            row = next((r for r in reports if r.get("variant") == name and "input_shape" in r), None)
            if row is None: continue
            shape = ",".join(map(str, row["input_shape"]))
            common = (f"--graph=/data/local/tmp/seamless-tpu/{name}.tflite "
                      f"--input_layer=activation --input_layer_shape={shape} "
                      f"--input_layer_value_files=activation:/data/local/tmp/seamless-tpu/{name}.input.bin "
                      "--num_threads=4 --warmup_runs=10 --num_runs=100")
            for backend, flags in (
                ("CPU", "--use_nnapi=false --use_xnnpack=true"),
                ("TPU", "--use_nnapi=true --nnapi_accelerator_name=google-edgetpu "
                 "--disable_nnapi_cpu=true --use_xnnpack=false --require_full_delegation=true"),
            ):
                commands.extend([
                    f"# {name} / {backend}", "adb logcat -c",
                    f"""adb shell am start -S -n org.tensorflow.lite.benchmark/.BenchmarkModelActivity --es args '"{common} {flags}"'""",
                    "# Wait for compile and all measured iterations before retrieving logs.",
                    "sleep 20", f"adb logcat -d > {name}-{backend.lower()}.log",
                ])
    (out / "BENCHMARK-COMMANDS.txt").write_text("\n".join(commands) + "\n")
    (out / "README.txt").write_text(
        "Real SeamlessStreaming decoder projection probe; Meta SeamlessStreaming CC BY-NC 4.0. "
        "Static-shape subgraph only, not a complete stage. FP16 uses FP16 input, weights and output. "
        "INT8-pt disables per-channel quantization for FULLY_CONNECTED weights and asserts one weight scale. "
        "These are targeted compatibility hypotheses based on the Pixel logs, not presumed fixes. "
        "Require full delegation, successful google-edgetpu compilation, completed CPU/TPU timings and output parity. "
        "INT8 calibration uses one Mandarin fixture only. Do not deploy.\n")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("source"); p.add_argument("output")
    a = p.parse_args(); convert(a.source, a.output)
