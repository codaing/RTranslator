"""Convert real decoder FFN captures into row-split grouped INT8 TFLite models.

The M=32 prefill capture contains one exceptional first row. Build a separate
M=1 model for that row and M=31 models for the remaining rows, so one row
cannot set the shared activation scale for all 32 rows. Sweep final output
group sizes on the M=31 tail. No production integration is performed here.
"""
import argparse
import json
import shlex
from pathlib import Path

import numpy as np
import tensorflow as tf
from scipy.optimize import linear_sum_assignment

from compare_tpu_ffn_outputs import (
    group_error_metrics,
    row_error_metrics,
    split_prefill_rows,
)
from convert_tpu_bank import metrics


CHANNELS = 1024
PRECISION_THRESHOLD = 0.05


def _load_capture(source):
    source = Path(source)
    meta = json.loads((source / "plan.json").read_text())
    with np.load(source / "weights.npz", allow_pickle=False) as archive:
        weights = {
            name: archive[key] for name, key in meta["constants"].items()
        }
    cases = []
    for item in meta["cases"]:
        with np.load(source / item["file"], allow_pickle=False) as archive:
            cases.append((
                archive["x"].astype(np.float32),
                archive["y"].astype(np.float32),
            ))
    return meta, weights, cases


def _build_workloads(cases, first_group_sizes, tail_group_sizes):
    full = [(x, y) for x, y in cases if len(x) == 32]
    if len(full) < 2:
        raise ValueError("need at least two real M=32 prefill calls")

    first_rows = []
    tail_rows = []
    for x, y in full:
        (x0, y0), (xt, yt) = split_prefill_rows(x, y)
        first_rows.append((x0, y0))
        tail_rows.append((xt, yt))

    return [
        ("m32-full", full, [32], 0),
        ("m1-prefill-first", first_rows, first_group_sizes, 0),
        ("m31-prefill-tail", tail_rows, tail_group_sizes, 1),
    ]


def _build_model(meta, weights, rows, group_size):
    shape = (1, 1, rows, CHANNELS)
    output_groups = CHANNELS // group_size

    class GroupedFFN(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape, tf.float32, name="activation")
            ]
        )
        def __call__(self, activation):
            values = {name: tf.constant(value) for name, value in weights.items()}
            values[meta["input"]] = activation
            matmul_index = 0

            for step in meta["steps"]:
                op = step["op"]
                args = [values[name] for name in step["inputs"]]
                if op == "MatMul":
                    left, right = args
                    if matmul_index == 0:
                        result = tf.nn.conv2d(
                            left,
                            tf.reshape(right, [1, 1, CHANNELS, 8192]),
                            1,
                            "VALID",
                        )
                    elif matmul_index == 1:
                        pieces = []
                        for start in range(0, CHANNELS, group_size):
                            filt = tf.reshape(
                                right[:, start:start + group_size],
                                [1, 1, 8192, group_size],
                            )
                            pieces.append(tf.nn.conv2d(left, filt, 1, "VALID"))
                        result = pieces
                    else:
                        raise ValueError("unexpected extra FFN MatMul")
                    matmul_index += 1
                elif op == "Add":
                    left, right = args
                    if isinstance(left, list) or isinstance(right, list):
                        pieces = left if isinstance(left, list) else right
                        bias = right if isinstance(left, list) else left
                        if len(pieces) != output_groups:
                            raise ValueError("wrong grouped output count")
                        result = [
                            pieces[index]
                            + bias[index * group_size:(index + 1) * group_size]
                            for index in range(len(pieces))
                        ]
                    else:
                        result = tf.add(left, right)
                elif op == "Relu":
                    value = args[0]
                    result = (
                        [tf.nn.relu(piece) for piece in value]
                        if isinstance(value, list)
                        else tf.nn.relu(value)
                    )
                elif op in ("Identity", "Cast"):
                    result = args[0]
                else:
                    raise ValueError("unsupported FFN op: " + op)
                values[step["output"]] = result

            output = values[meta["output"]]
            if not isinstance(output, list) or len(output) != output_groups:
                raise ValueError("FFN did not produce expected independent groups")
            return tuple(output)

    module = GroupedFFN()
    concrete = module.__call__.get_concrete_function()
    return module, concrete


def _expansion_reference(meta, weights, x):
    first_matmul = next(step for step in meta["steps"] if step["op"] == "MatMul")
    first_add = next(
        step for step in meta["steps"]
        if step["op"] == "Add" and first_matmul["output"] in step["inputs"]
    )
    weight_name = first_matmul["inputs"][1]
    bias_name = next(
        name for name in first_add["inputs"] if name != first_matmul["output"]
    )
    return np.maximum(x @ weights[weight_name] + weights[bias_name], 0.0)


def _expansion_quantization_report(
    interpreter, tensor_details, expansion_tensor_index, meta, weights, x
):
    detail = tensor_details[expansion_tensor_index]
    if detail["dtype"] != np.int8:
        raise ValueError("expansion intermediate is not INT8")
    scale, zero = detail["quantization"]
    if scale <= 0:
        raise ValueError("invalid expansion activation quantization")
    raw = interpreter.get_tensor(expansion_tensor_index).reshape(len(x), 8192)
    dequantized = (raw.astype(np.float32) - int(zero)) * float(scale)
    reference = _expansion_reference(meta, weights, x)
    if reference.shape != dequantized.shape:
        raise ValueError("expansion intermediate shape mismatch")

    channel_norms = np.linalg.norm(reference.astype(np.float64), axis=0)
    active_channels = channel_norms > max(float(np.max(channel_norms)) * 1e-6, 1e-8)
    channel_error = np.linalg.norm(
        (dequantized - reference).astype(np.float64)[:, active_channels], axis=0
    ) / channel_norms[active_channels]
    return {
        "scale": float(scale),
        "zero_point": int(zero),
        "vs_fp32": metrics(reference, dequantized),
        "active_channel_count": int(np.count_nonzero(active_channels)),
        "inactive_channel_count": int(np.count_nonzero(~active_channels)),
        "channel_nrms_p50_active": float(np.percentile(channel_error, 50)),
        "channel_nrms_p95_active": float(np.percentile(channel_error, 95)),
        "channel_nrms_p99_active": float(np.percentile(channel_error, 99)),
        "channel_nrms_max_active": float(np.max(channel_error)),
        "zero_fraction": float(np.mean(raw == int(zero))),
        "positive_saturation_fraction": float(np.mean(raw == 127)),
    }


def _convert_variant(
    output_dir,
    meta,
    weights,
    workload_name,
    calibration_cases,
    evaluation_case,
    source_row_offset,
    group_size,
):
    x, y = evaluation_case
    rows = len(x)
    output_groups = CHANNELS // group_size
    shape = (1, 1, rows, CHANNELS)

    def reshape(value):
        return np.ascontiguousarray(value.reshape(shape), dtype=np.float32)

    calibration = [a for a, _ in calibration_cases]
    if len(calibration) > 32:
        indexes = np.linspace(0, len(calibration) - 1, 32, dtype=int)
        calibration = [calibration[index] for index in indexes]

    variant = f"ffn-m{rows}-{workload_name}-g{group_size}-int8"
    report = {
        "variant": variant,
        "workload": workload_name,
        "rows": rows,
        "source_row_offset": source_row_offset,
        "group_size": group_size,
        "output_groups": output_groups,
        "calibration_calls": len(calibration),
        "calibration_rows": int(sum(len(value) for value in calibration)),
        "status": "FAIL",
    }

    try:
        module, concrete = _build_model(meta, weights, rows, group_size)
        fp32_parts = [
            value.numpy().reshape(rows, group_size)
            for value in module(reshape(x))
        ]
        fp32_output = np.concatenate(fp32_parts, axis=1)
        fp32_error = metrics(y, fp32_output)
        report["fp32_vs_source_fp16"] = fp32_error
        if fp32_error["nrms"] > 0.02:
            raise ValueError("FP32 grouped lowering differs from original FP16 core")


        if workload_name == "m1-prefill-first" and group_size == 1024:
            try:
                fp32_converter = tf.lite.TFLiteConverter.from_concrete_functions(
                    [concrete], module
                )
                fp32_model_blob = fp32_converter.convert()
                fp32_interpreter = tf.lite.Interpreter(
                    model_content=fp32_model_blob,
                    num_threads=4,
                    experimental_op_resolver_type=(
                        tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
                    ),
                )
                fp32_interpreter.allocate_tensors()
                fp32_input_detail = fp32_interpreter.get_input_details()[0]
                fp32_output_details = fp32_interpreter.get_output_details()
                fp32_ops = fp32_interpreter._get_ops_details()
                if fp32_input_detail["dtype"] != np.float32:
                    raise ValueError("FP32 reference model input is not FLOAT32")
                if len(fp32_output_details) != 1 or fp32_output_details[0]["dtype"] != np.float32:
                    raise ValueError("FP32 reference model must have one FLOAT32 output")
                if any(op["op_name"] != "CONV_2D" for op in fp32_ops):
                    raise ValueError("FP32 reference model contains a non-Conv2D op")
                fp32_input_data = reshape(x)
                fp32_interpreter.set_tensor(
                    fp32_input_detail["index"], fp32_input_data
                )
                fp32_interpreter.invoke()
                fp32_actual = fp32_interpreter.get_tensor(
                    fp32_output_details[0]["index"]
                ).reshape(rows, CHANNELS)
                fp32_actual_metrics = metrics(y, fp32_actual)
                fp32_actual_rows = row_error_metrics(y, fp32_actual)
                fp32_actual_groups = group_error_metrics(y, fp32_actual, 32)
                report.update({
                    "fp32_tflite_model_bytes": len(fp32_model_blob),
                    "fp32_tflite_input_shape": (
                        fp32_input_detail["shape"].tolist()
                    ),
                    "fp32_tflite_output_shape": (
                        fp32_output_details[0]["shape"].tolist()
                    ),
                    "fp32_tflite_ops": [op["op_name"] for op in fp32_ops],
                    "fp32_tflite_vs_source_fp16": fp32_actual_metrics,
                    "fp32_tflite_max_row_nrms": (
                        fp32_actual_rows["max_row_nrms"]
                    ),
                    "fp32_tflite_max_fixed_32_channel_group_nrms": (
                        fp32_actual_groups["max_group_nrms"]
                    ),
                })
                (output_dir / (variant + ".fp32.tflite")).write_bytes(
                    fp32_model_blob
                )
                (output_dir / (variant + ".fp32.input.bin")).write_bytes(
                    fp32_input_data.astype("<f4").tobytes()
                )
                (output_dir / (variant + ".fp32.cpu-output.bin")).write_bytes(
                    fp32_actual.astype("<f4").tobytes()
                )
            except Exception as fp32_error:
                report["fp32_tflite_error"] = str(fp32_error)

        converter = tf.lite.TFLiteConverter.from_concrete_functions(
            [concrete], module
        )
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = (
            lambda: iter([[reshape(value)] for value in calibration])
        )
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        model_blob = converter.convert()

        interpreter = tf.lite.Interpreter(
            model_content=model_blob,
            num_threads=4,
            experimental_op_resolver_type=(
                tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
            ),
        )
        interpreter.allocate_tensors()
        input_detail = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()
        operations = interpreter._get_ops_details()
        if len(operations) != 1 + output_groups:
            raise ValueError(
                f"expected {1 + output_groups} Conv2D operations, "
                f"found {len(operations)}"
            )
        if any(operation["op_name"] != "CONV_2D" for operation in operations):
            raise ValueError("grouped FFN contains a non-Conv2D operation")
        if len(output_details) != output_groups:
            raise ValueError("unexpected independent output tensor count")

        tensor_details = {
            item["index"]: item for item in interpreter.get_tensor_details()
        }
        filter_scale_counts = []
        for operation in operations:
            weight_tensor = tensor_details[int(operation["inputs"][1])]
            quant = weight_tensor["quantization_parameters"]
            if len(quant["scales"]) != int(weight_tensor["shape"][0]):
                raise ValueError("Conv2D filter is not per-output-channel quantized")
            if np.any(quant["zero_points"] != 0):
                raise ValueError("Conv2D weights are not symmetric")
            filter_scale_counts.append(int(len(quant["scales"])))
            for tensor_index in (*operation["inputs"], *operation["outputs"]):
                if tensor_index >= 0:
                    dtype = tensor_details[int(tensor_index)]["dtype"]
                    if dtype not in (np.int8, np.int32, np.int64):
                        raise ValueError("non-integer tensor in FFN graph")

        if input_detail["dtype"] != np.int8:
            raise ValueError("group model input is not INT8")
        if any(item["dtype"] != np.int8 for item in output_details):
            raise ValueError("group model output is not INT8")

        # Inspect the actual tensor consumed by the contraction Conv2D. This
        # is the post-bias/post-ReLU 8192-channel activation, not the earlier
        # expansion MatMul output (which has the same shape).
        contraction_ops = []
        for operation in operations:
            if operation["op_name"] != "CONV_2D":
                continue
            input_index, filter_index = map(int, operation["inputs"][:2])
            input_shape = tensor_details[input_index]["shape"].tolist()
            filter_shape = tensor_details[filter_index]["shape"].tolist()
            if (
                input_shape == [1, 1, rows, 8192]
                and filter_shape[1:] == [1, 1, 8192]
                and filter_shape[0] == group_size
            ):
                contraction_ops.append(operation)
        if len(contraction_ops) != output_groups:
            raise ValueError(
                f"expected {output_groups} grouped contraction Conv2D ops, "
                f"found {len(contraction_ops)}"
            )
        expansion_tensor_index = int(contraction_ops[0]["inputs"][0])

        input_scale, input_zero = input_detail["quantization"]
        if input_scale <= 0:
            raise ValueError("invalid input quantization")
        input_data = np.clip(
            np.rint(reshape(x) / input_scale) + input_zero,
            -128,
            127,
        ).astype(np.int8)
        interpreter.set_tensor(input_detail["index"], input_data)
        interpreter.invoke()

        expansion_report = _expansion_quantization_report(
            interpreter,
            tensor_details,
            expansion_tensor_index,
            meta,
            weights,
            x,
        )

        parts = []
        raw_parts = []
        output_quantizations = []
        output_names = []
        for detail in output_details:
            raw = interpreter.get_tensor(detail["index"])
            output_scale, output_zero = detail["quantization"]
            parts.append(
                ((raw.astype(np.float32) - output_zero) * output_scale)
                .reshape(rows, group_size)
            )
            raw_parts.append(raw.tobytes())
            output_quantizations.append([
                float(output_scale), int(output_zero)
            ])
            output_names.append(detail["name"])

        cost = np.empty((len(parts), output_groups), dtype=np.float64)
        for tensor_index, part in enumerate(parts):
            for group_index in range(output_groups):
                reference_group = y[
                    :, group_index * group_size:(group_index + 1) * group_size
                ]
                cost[tensor_index, group_index] = (
                    np.linalg.norm(part - reference_group)
                    / max(np.linalg.norm(reference_group), 1e-12)
                )

        tensor_indexes, group_indexes = linear_sum_assignment(cost)
        group_to_tensor = [-1] * output_groups
        for tensor_index, group_index in zip(tensor_indexes, group_indexes):
            group_to_tensor[int(group_index)] = int(tensor_index)
        if (
            any(index < 0 for index in group_to_tensor)
            or len(set(group_to_tensor)) != len(parts)
        ):
            raise ValueError("could not establish unique TFLite output ordering")

        actual = np.concatenate(
            [parts[index] for index in group_to_tensor], axis=1
        )
        global_error = metrics(y, actual)
        group_error = group_error_metrics(y, actual, 32)
        quantization_group_error = group_error_metrics(y, actual, group_size)
        row_error = row_error_metrics(y, actual)

        report.update({
            "status": "CONVERTED",
            "model_bytes": len(model_blob),
            "input_shape": input_detail["shape"].tolist(),
            "output_shapes": [
                item["shape"].tolist() for item in output_details
            ],
            "input_name": input_detail["name"],
            "input_quantization": [
                float(input_scale), int(input_zero)
            ],
            "output_names": output_names,
            "output_quantizations": output_quantizations,
            "output_group_to_tensor_index": group_to_tensor,
            "filter_scale_counts": filter_scale_counts,
            "ops": [operation["op_name"] for operation in operations],
            "expansion_activation": expansion_report,
            "nrms": global_error["nrms"],
            "max_abs": global_error["max_abs"],
            "fixed_32_channel_group_nrms": group_error["group_nrms"],
            "max_fixed_32_channel_group_nrms": group_error["max_group_nrms"],
            "p95_fixed_32_channel_group_nrms": group_error["p95_group_nrms"],
            "fixed_32_channel_groups_over_5pct": group_error["groups_over_5pct"],
            "quantization_group_nrms": quantization_group_error["group_nrms"],
            "max_quantization_group_nrms": quantization_group_error["max_group_nrms"],
            "p95_quantization_group_nrms": quantization_group_error["p95_group_nrms"],
            "row_nrms": row_error["row_nrms"],
            "max_row_nrms": row_error["max_row_nrms"],
            "p95_row_nrms": row_error["p95_row_nrms"],
            "worst_row_index": row_error["worst_row_index"],
            "worst_source_row_index": (
                source_row_offset + row_error["worst_row_index"]
            ),
            "global_precision_screen_pass": (
                global_error["nrms"] < PRECISION_THRESHOLD
            ),
            "group_precision_screen_pass": (
                group_error["max_group_nrms"] < PRECISION_THRESHOLD
            ),
            "row_precision_screen_pass": (
                row_error["max_row_nrms"] < PRECISION_THRESHOLD
            ),
            "precision_screen_pass": (
                global_error["nrms"] < PRECISION_THRESHOLD
                and group_error["max_group_nrms"] < PRECISION_THRESHOLD
                and row_error["max_row_nrms"] < PRECISION_THRESHOLD
            ),
            "precision_threshold": PRECISION_THRESHOLD,
        })

        (output_dir / (variant + ".tflite")).write_bytes(model_blob)
        (output_dir / (variant + ".input.bin")).write_bytes(
            input_data.tobytes()
        )
        (output_dir / (variant + ".cpu-output.bin")).write_bytes(
            b"".join(raw_parts)
        )
        (output_dir / (variant + ".source-reference.bin")).write_bytes(
            y.astype("<f4").tobytes()
        )
    except Exception as error:
        report["status"] = "FAIL"
        report["error"] = str(error)

    print(json.dumps(report), flush=True)
    return report


def convert(
    source,
    output,
    first_group_sizes=(64, 128, 256, 512, 1024),
    tail_group_sizes=(8, 16, 32, 64),
):
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta, weights, cases = _load_capture(source)
    first_group_sizes = list(dict.fromkeys(int(value) for value in first_group_sizes))
    tail_group_sizes = list(dict.fromkeys(int(value) for value in tail_group_sizes))
    if not first_group_sizes:
        raise ValueError("at least one first-row group size is required")
    if not tail_group_sizes:
        raise ValueError("at least one tail group size is required")
    for group_size in first_group_sizes + tail_group_sizes:
        if group_size <= 0 or CHANNELS % group_size:
            raise ValueError(f"invalid output group size: {group_size}")

    reports = []
    for workload_name, workload_cases, group_sizes, row_offset in _build_workloads(
        cases, first_group_sizes, tail_group_sizes
    ):
        evaluation_case = workload_cases[-1]
        calibration_cases = workload_cases[:-1]
        if not calibration_cases:
            raise ValueError(f"not enough calibration calls for {workload_name}")
        for group_size in group_sizes:
            report = _convert_variant(
                output_dir,
                meta,
                weights,
                workload_name,
                calibration_cases,
                evaluation_case,
                row_offset,
                group_size,
            )
            reports.append(report)
            (output_dir / "manifest.json").write_text(
                json.dumps(reports, indent=2) + "\n"
            )

    app_output = "/sdcard/Android/data/org.tensorflow.lite.benchmark/files"
    commands = [
        "# Run from the extracted artifact directory",
        "adb shell mkdir -p /data/local/tmp/seamless-tpu",
        f"adb shell mkdir -p {app_output}",
        "adb push . /data/local/tmp/seamless-tpu/",
    ]
    for report in reports:
        if report["status"] != "CONVERTED":
            continue
        name = report["variant"]
        shape = ",".join(map(str, report["input_shape"]))
        common = (
            f"--graph=/data/local/tmp/seamless-tpu/{name}.tflite "
            f"--input_layer=activation --input_layer_shape={shape} "
            f"--input_layer_value_files=activation:/data/local/tmp/seamless-tpu/"
            f"{name}.input.bin --num_threads=4 --warmup_runs=10 --num_runs=100"
        )
        # Run XNNPACK first so the TPU comparison can use the matching
        # physical-device CPU output, rather than the CI-host reference.
        for backend, flags in (
            ("cpu", "--use_nnapi=false --use_xnnpack=true"),
            (
                "tpu",
                "--use_nnapi=true "
                "--nnapi_accelerator_name=google-edgetpu "
                "--disable_nnapi_cpu=true --use_xnnpack=false "
                "--require_full_delegation=true",
            ),
        ):
            output_path = f"{app_output}/{name}-{backend}-output.bin"
            expected_bytes = int(report["rows"]) * CHANNELS
            commands.extend([
                "",
                f"# {name} / {backend}",
                "adb logcat -c",
                f"adb shell rm -f {output_path}",
                (
                    "adb shell am start -S -n "
                    "org.tensorflow.lite.benchmark/.BenchmarkModelActivity "
                    "--es args "
                    + shlex.quote(
                        f'"{common} {flags} --output_filepath={output_path}"'
                    )
                ),
                "# Wait up to 180 seconds for the Activity to write raw output.",
                (
                    f"for i in $(seq 1 180); do "
                    f"if adb shell test -s {output_path}; then break; fi; "
                    "sleep 1; done"
                ),
                "sleep 20",
               (
                    "adb logcat -d -s tflite_BenchmarkModelActivity "
                    f"tflite ExecutionPlan Darwinn > {name}-{backend}.log"
                ),
                (
                    f"adb shell test -s {output_path} || "
                    f"{{ echo 'ERROR: benchmark output missing; inspect "
                    f"{name}-{backend}.log'; exit 1; }}"
                ),
                (
                    f"bytes=$(adb shell wc -c {output_path} | awk '{{print $1}}' | tr -d '\\r'); "
                    f"test $bytes = {expected_bytes} || "
                    f"{{ echo 'ERROR: expected {expected_bytes} bytes, got' "
                    "$bytes; exit 1; }"
                ),
                f"adb pull {output_path} {name}-{backend}-output.bin",
                (
                    f"python compare_tpu_ffn_outputs.py manifest.json {name} "
                    f"{name}-{backend}-output.bin"
                    + (
                        f" --host-output {name}-cpu-output.bin "
                        "--require-exact-backend-parity"
                        if backend == "tpu" else ""
                    )
                    + " || echo 'Precision/parity gate failed; retain metrics "
                    f"and continue: {name}/{backend}'"
                ),
            ])

    # Benchmark the first row in FP32 on CPU. The tail remains INT8 and
    # is measured separately on both XNNPACK and google-edgetpu.
    fp32_first = next(
        (
            report for report in reports
            if report["status"] == "CONVERTED"
            and report["workload"] == "m1-prefill-first"
            and report["group_size"] == 1024
            and report.get("fp32_tflite_model_bytes")
        ),
        None,
    )
    if fp32_first is not None:
        name = fp32_first["variant"]
        shape = ",".join(map(str, fp32_first["fp32_tflite_input_shape"]))
        common = (
            f"--graph=/data/local/tmp/seamless-tpu/{name}.fp32.tflite "
            f"--input_layer=activation --input_layer_shape={shape} "
            f"--input_layer_value_files=activation:/data/local/tmp/seamless-tpu/"
            f"{name}.fp32.input.bin --num_threads=4 --warmup_runs=10 --num_runs=100"
        )
        output_path = f"{app_output}/{name}-fp32-cpu-output.bin"
        expected_bytes = int(fp32_first["rows"]) * CHANNELS * 4
        commands.extend([
            "",
            f"# {name} FP32 first row / CPU XNNPACK",
            "adb logcat -c",
            f"adb shell rm -f {output_path}",
            (
                "adb shell am start -S -n "
                "org.tensorflow.lite.benchmark/.BenchmarkModelActivity "
                "--es args "
                + shlex.quote(
                    f'"{common} --use_nnapi=false --use_xnnpack=true '
                    f'--output_filepath={output_path}"'
                )
            ),
            "# Wait up to 180 seconds for the Activity to write raw output.",
            (
                f"for i in $(seq 1 180); do "
                f"if adb shell test -s {output_path}; then break; fi; "
                "sleep 1; done"
            ),
            "sleep 20",
            (
                "adb logcat -d -s tflite_BenchmarkModelActivity "
                f"tflite ExecutionPlan Darwinn > {name}-fp32-cpu.log"
            ),
            (
                f"adb shell test -s {output_path} || "
                f"{{ echo 'ERROR: FP32 output missing; inspect {name}-fp32-cpu.log'; exit 1; }}"
            ),
            (
                f"bytes=$(adb shell wc -c {output_path} | awk '{{print $1}}' | tr -d '\\r'); "
                f"test $bytes = {expected_bytes} || "
                f"{{ echo 'ERROR: expected {expected_bytes} bytes, got' $bytes; exit 1; }}"
            ),
            f"adb pull {output_path} {name}-fp32-cpu-output.bin",
            (
                "python compare_tpu_ffn_float_output.py manifest.json "
                f"{name} {name}-fp32-cpu-output.bin "
                f"--host-output {name}.fp32.cpu-output.bin "
                "|| echo 'FP32 first-row precision screen failed; inspect metrics'"
            ),
        ])

    # Reassemble the split first-row + tail result and compare with
    # the single M=32 baseline for each physical backend.
    for backend in ("cpu", "tpu"):
        for first_group_size in first_group_sizes:
            for tail_group_size in tail_group_sizes:
                commands.extend([
                    "",
                    (
                        f"# Reassembled M=1(g{first_group_size}) + "
                        f"M=31(g{tail_group_size}) / {backend}"
                    ),
                    (
                        "python aggregate_tpu_ffn_split.py manifest.json "
                        f"--backend {backend} "
                        f"--first-group-size {first_group_size} "
                        f"--tail-group-size {tail_group_size} "
                        f"--output SPLIT-ACCURACY-{backend}-first-g{first_group_size}"
                        f"-tail-g{tail_group_size}.json"
                        " || echo 'Split accuracy audit failed; inspect the JSON/logs'"
                    ),
                ])

    if fp32_first is not None:
        for backend in ("cpu", "tpu"):
            for tail_group_size in tail_group_sizes:
                commands.extend([
                    "",
                    (
                        f"# Mixed FP32 CPU first row + INT8 M=31 tail / "
                        f"{backend} / tail group {tail_group_size}"
                    ),
                    (
                        "python aggregate_tpu_ffn_split.py manifest.json "
                        f"--backend {backend} --first-group-size 1024 "
                        f"--tail-group-size {tail_group_size} "
                        f"--first-fp32-output {fp32_first['variant']}-fp32-cpu-output.bin "
                        f"--output MIXED-ACCURACY-{backend}-tail-g{tail_group_size}.json"
                        " || echo 'Mixed precision audit failed; inspect metrics'"
                    ),
                ])

    (output_dir / "BENCHMARK-COMMANDS.txt").write_text(
        "\n".join(commands) + "\n"
    )
    (output_dir / "README.txt").write_text(
        "Experimental real decoder FFN row-split quantization sweep. "
        "The M=32 prefill batch is evaluated as: (1) one M=32 baseline, "
        "(2) a separate M=1 model for the first row, and (3) an M=31 model "
        "for rows 1..31. Both split segments sweep output group sizes 8, 16, "
        "32 and 64 for the M=31 tail; M=1 first-row groups are 64, 128, "
        "256, 512 and 1024. Each segment is calibrated independently. "
        "The first row is calibrated only from first rows of the "
        "captured M=32 calls; the tail is calibrated only from the remaining "
        "31 rows. Each manifest entry reports global, per-row and per-output-"
        "group errors, plus the actual quantized expansion intermediate. "
        "A separate FP32 M=1 first-row model is also emitted for CPU XNNPACK; "
        "the mixed candidate is FP32 CPU first row plus INT8 TPU M=31 tail. "
        "Use fixed 32-channel evaluation windows independent of quantization "
        "group size. Require full google-edgetpu delegation, exact CPU/TPU "
        "raw output parity, and all accuracy gates before considering any model "
        "further. Same-recording calibration/evaluation only; no held-out "
        "speech, whole-decoder or full-pipeline quality claim. Do not deploy. "
        "Meta Seamless model materials are CC BY-NC 4.0; respect the "
        "noncommercial restriction.\n"
    )
    if not any(report["status"] == "CONVERTED" for report in reports):
        raise RuntimeError("all row-split grouped FFN conversions failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument(
        "--first-group-sizes",
        default="64,128,256,512,1024",
        help="comma-separated output group sizes for the M=1 first row",
    )
    parser.add_argument(
        "--tail-group-sizes",
        default="8,16,32,64",
        help="comma-separated output group sizes for the M=31 tail",
    )
    args = parser.parse_args()
    first_sizes = [
        int(value.strip()) for value in args.first_group_sizes.split(",")
        if value.strip()
    ]
    tail_sizes = [
        int(value.strip()) for value in args.tail_group_sizes.split(",")
        if value.strip()
    ]
    convert(args.source, args.output, first_sizes, tail_sizes)
