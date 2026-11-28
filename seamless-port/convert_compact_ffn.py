#!/usr/bin/env python3
"""Experimental compact FP32 first-row FFN; one layer, one recording only."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf

from compare_tpu_ffn_outputs import nrms, group_error_metrics

WIDTHS = (448, 512, 1024)


def load(source):
    plan = json.loads((source / "plan.json").read_text())
    steps = plan["steps"]
    if [s["op"] for s in steps] != ["MatMul", "Add", "Relu", "MatMul", "Add"]:
        raise ValueError("unexpected FFN topology")
    with np.load(source / "weights.npz", allow_pickle=False) as z:
        weights = {name: z[key].astype(np.float32)
                   for name, key in plan["constants"].items()}
    w1 = weights[steps[0]["inputs"][1]]
    b1 = weights[next(n for n in steps[1]["inputs"] if n in weights)]
    w2 = weights[steps[3]["inputs"][1]]
    b2 = weights[next(n for n in steps[4]["inputs"] if n in weights)]
    if (w1.shape, b1.shape, w2.shape, b2.shape) != (
        (1024, 8192), (8192,), (8192, 1024), (1024,)
    ):
        raise ValueError("unexpected FFN dimensions")
    cases = []
    for item in plan["cases"]:
        if item["rows"] < 2:
            continue
        with np.load(source / item["file"], allow_pickle=False) as z:
            cases.append((item["file"], item["rows"],
                          z["x"][:1].astype(np.float32),
                          z["y"][:1].astype(np.float32)))
    return plan, (w1, b1, w2, b2), cases


def evaluate(reference, output):
    if not np.isfinite(output).all():
        raise ValueError("nonfinite inference output")
    return {"nrms": nrms(reference, output),
            "max_fixed32_nrms": group_error_metrics(reference, output, 32)["max_group_nrms"],
            "max_abs": float(np.max(np.abs(reference - output)))}


def convert(source, dest):
    dest.mkdir(parents=True, exist_ok=True)
    plan, (w1, b1, w2, b2), cases = load(source)
    indexes = [i for i, c in enumerate(cases) if c[1] == 32]
    if len(indexes) != 8:
        raise ValueError("expected eight M=32 calls from this capture")
    train, test = indexes[:-1], indexes[-1]
    x = np.concatenate([c[2] for c in cases], axis=0)
    y = np.concatenate([c[3] for c in cases], axis=0)
    # Only calibration calls select channels; neither target nor other
    # diagnostic calls influence ranking. Stable sort makes ties deterministic.
    pre_train = x[train] @ w1 + b1
    h_train = np.maximum(pre_train, 0)
    score = np.max(h_train, axis=0) * np.linalg.norm(w2, axis=1)
    # Pad with inactive channels nearest to their activation boundary, not
    # arbitrary zero-score ties. All ranking inputs come from calibration.
    weight_norm = np.maximum(np.linalg.norm(w1, axis=0), 1e-12)
    margin_score = np.max(pre_train / weight_norm, axis=0)
    ranking = np.lexsort((np.arange(len(score)), -margin_score, -score))
    pre_all = x @ w1 + b1
    report = {
        "source_graph_sha256": plan["source_graph_sha256"],
        "calibration_cases": [cases[i][0] for i in train],
        "evaluation_case": cases[test][0],
        "active_calibration_channels": int(np.count_nonzero(score)),
        "ranking": "active_contribution_then_normalized_preactivation_margin",
        "fixed_eval_group_size": 32,
        "threshold": 0.05,
        "same_recording_only": True,
        "variants": [],
    }
    (dest / "first-row.input.bin").write_bytes(x[test].astype("<f4").tobytes())
    (dest / "first-row.source-reference.bin").write_bytes(y[test].astype("<f4").tobytes())
    for k in WIDTHS:
        ids = np.sort(ranking[:k])
        omitted = np.ones(8192, dtype=bool)
        omitted[ids] = False
        # Real-arithmetic L2 certificate: every omitted preactivation
        # remains negative if distance from a calibration anchor is less
        # than its minimum normalized negative margin. Reserve a small
        # safety margin; this is a diagnostic, not a deployed guard.
        radii = np.maximum(0, np.min(-pre_train[:, omitted] / weight_norm[omitted], axis=1) - 1e-4)
        f1 = np.ascontiguousarray(w1[:, ids].reshape(1, 1, 1024, k))
        f2 = np.ascontiguousarray(w2[ids, :].reshape(1, 1, k, 1024))
        bias1 = np.ascontiguousarray(b1[ids])
        class CompactFFN(tf.Module):
            @tf.function(input_signature=[tf.TensorSpec([1, 1, 1, 1024], tf.float32, name="activation")])
            def __call__(self, activation):
                h = tf.nn.relu(tf.nn.conv2d(activation, tf.constant(f1), 1, "VALID") + tf.constant(bias1))
                return tf.nn.conv2d(h, tf.constant(f2), 1, "VALID") + tf.constant(b2)

        module = CompactFFN()
        concrete = module.__call__.get_concrete_function()
        blob = tf.lite.TFLiteConverter.from_concrete_functions([concrete], module).convert()
        interpreter = tf.lite.Interpreter(model_content=blob, num_threads=4,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
        interpreter.allocate_tensors()
        inp = interpreter.get_input_details()[0]
        out = interpreter.get_output_details()[0]
        ops = [op["op_name"] for op in interpreter._get_ops_details()]
        if ops != ["CONV_2D", "CONV_2D"]:
            raise ValueError(f"unexpected operations: {ops}")
        for tensor in (inp, out):
            if tensor["dtype"] != np.float32 or tensor["shape"].tolist() != [1, 1, 1, 1024]:
                raise ValueError("unexpected boundary type or shape")
        per_case = []
        name = f"ffn-m1-compact-k{k}-fp32"
        for i, case in enumerate(cases):
            interpreter.set_tensor(inp["index"], x[i].reshape(1, 1, 1, 1024))
            interpreter.invoke()
            actual = interpreter.get_tensor(out["index"]).reshape(1, 1024)
            metric = evaluate(y[i:i+1], actual)
            per_case.append({"case": case[0], "rows": case[1],
                             "role": "calibration" if i in train else "evaluation" if i == test else "same_recording_diagnostic",
                             "omitted_active_channels": int(np.count_nonzero(pre_all[i, omitted] > 0)),
                             "l2_inactive_certificate": bool(np.any(np.linalg.norm(x[i] - x[train], axis=1) < radii)),
                             **metric})
            if i == test:
                (dest / (name + ".host-output.bin")).write_bytes(actual.astype("<f4").tobytes())
        worst_nrms = max(v["nrms"] for v in per_case)
        worst_group = max(v["max_fixed32_nrms"] for v in per_case)
        result = {"variant": name, "hidden_channels": k, "model_bytes": len(blob),
                  "ops": ops, "selected_indices": ids.tolist(), "per_case": per_case,
                  "anchor_radii_l2": radii.tolist(),
                  "max_omitted_active_channels": max(v["omitted_active_channels"] for v in per_case),
                  "evaluation": per_case[test], "worst_nrms": worst_nrms,
                  "worst_fixed32_nrms": worst_group,
                  "precision_screen_pass": worst_nrms < 0.05 and worst_group < 0.05}
        (dest / (name + ".tflite")).write_bytes(blob)
        report["variants"].append(result)
        print(json.dumps({key: value for key, value in result.items() if key not in ("selected_indices", "per_case")}), flush=True)
    (dest / "compact-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    shutil.copyfile(source / "MODEL-LICENSE.txt", dest / "MODEL-LICENSE.txt")
    if not all(v["precision_screen_pass"] for v in report["variants"]):
        raise RuntimeError("compact FP32 precision screen failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("dest", type=Path)
    args = parser.parse_args()
    convert(args.source, args.dest)
