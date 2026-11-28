#!/usr/bin/env python3
"""Reassemble M=1 + M=31 FFN outputs, including mixed FP32/INT8 execution."""
import argparse
import json
from pathlib import Path

import numpy as np

from compare_tpu_ffn_outputs import (
    group_error_metrics,
    nrms,
    reconstruct,
    row_error_metrics,
)

EVAL_GROUP_SIZE = 32


def load_output(root, meta, backend):
    path = root / f"{meta['variant']}-{backend}-output.bin"
    return reconstruct(path, meta), path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--backend", choices=("cpu", "tpu"), required=True)
    parser.add_argument("--first-group-size", type=int, required=True)
    parser.add_argument("--tail-group-size", type=int, required=True)
    parser.add_argument("--first-fp32-output", default=None)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    root = manifest_path.parent
    rows = json.loads(manifest_path.read_text())

    def choose(workload, group_size=None):
        matches = [
            item for item in rows
            if item.get("status") == "CONVERTED"
            and item.get("workload") == workload
            and (group_size is None or int(item["group_size"]) == group_size)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one converted {workload} model "
                f"for group size {group_size}, found {len(matches)}"
            )
        return matches[0]

    full_meta = choose("m32-full", 32)
    first_meta = choose("m1-prefill-first", args.first_group_size)
    tail_meta = choose("m31-prefill-tail", args.tail_group_size)

    full, full_path = load_output(root, full_meta, args.backend)
    tail, tail_path = load_output(root, tail_meta, args.backend)

    if args.first_fp32_output:
        first_path = Path(args.first_fp32_output)
        if not first_path.is_absolute():
            first_path = root / first_path
        first_raw = np.fromfile(first_path, dtype="<f4")
        expected = int(first_meta["rows"]) * 1024
        if first_raw.size != expected:
            raise ValueError(
                f"{first_path}: {first_raw.size} float32 values, expected {expected}"
            )
        first = first_raw.reshape(int(first_meta["rows"]), 1024)
        first_mode = "fp32_cpu"
    else:
        first, first_path = load_output(root, first_meta, args.backend)
        first_mode = "int8"

    full_ref = np.fromfile(
        root / (full_meta["variant"] + ".source-reference.bin"),
        dtype="<f4",
    ).reshape(32, 1024)
    first_ref = np.fromfile(
        root / (first_meta["variant"] + ".source-reference.bin"),
        dtype="<f4",
    ).reshape(1, 1024)
    tail_ref = np.fromfile(
        root / (tail_meta["variant"] + ".source-reference.bin"),
        dtype="<f4",
    ).reshape(31, 1024)
    split_ref = np.concatenate((first_ref, tail_ref), axis=0)
    if not np.array_equal(split_ref, full_ref):
        raise ValueError("M=1 and M=31 references do not reassemble to M=32")

    split = np.concatenate((first, tail), axis=0)
    full_rows = row_error_metrics(full_ref, full)
    split_rows = row_error_metrics(full_ref, split)

    # Keep evaluation windows fixed at 32 channels for every model. Report
    # quantization-group error separately; never let a larger quantization
    # group make the acceptance window larger.
    full_eval_groups = group_error_metrics(
        full_ref, full, EVAL_GROUP_SIZE
    )
    first_eval_groups = group_error_metrics(
        first_ref, first, EVAL_GROUP_SIZE
    )
    tail_eval_groups = group_error_metrics(
        tail_ref, tail, EVAL_GROUP_SIZE
    )
    first_quant_groups = (
        group_error_metrics(first_ref, first, args.first_group_size)
        if first_mode == "int8" else None
    )
    tail_quant_groups = group_error_metrics(
        tail_ref, tail, args.tail_group_size
    )

    result = {
        "backend_for_m32_and_tail": args.backend,
        "first_mode": first_mode,
        "first_group_size_reference_variant": args.first_group_size,
        "tail_group_size": args.tail_group_size,
        "fixed_evaluation_group_size": EVAL_GROUP_SIZE,
        "source_rows": 32,
        "full_m32": {
            "model": full_meta["variant"],
            "output": str(full_path),
            "global_nrms": nrms(full_ref, full),
            "max_row_nrms": full_rows["max_row_nrms"],
            "p95_row_nrms": full_rows["p95_row_nrms"],
            "worst_row_index": full_rows["worst_row_index"],
            "max_fixed_32_channel_group_nrms": full_eval_groups["max_group_nrms"],
        },
        "split_m1_plus_m31": {
            "first_model": (
                first_meta["variant"] + " (FP32 CPU)"
                if first_mode == "fp32_cpu" else first_meta["variant"]
            ),
            "first_output": str(first_path),
            "tail_model": tail_meta["variant"],
            "tail_output": str(tail_path),
            "global_nrms": nrms(full_ref, split),
            "max_abs": float(np.max(np.abs(full_ref - split))),
            "max_row_nrms": split_rows["max_row_nrms"],
            "p95_row_nrms": split_rows["p95_row_nrms"],
            "worst_row_index": split_rows["worst_row_index"],
            "first_fixed_32_channel_group_max_nrms": (
                first_eval_groups["max_group_nrms"]
            ),
            "tail_fixed_32_channel_group_max_nrms": (
                tail_eval_groups["max_group_nrms"]
            ),
            "first_quantization_group_max_nrms": (
                first_quant_groups["max_group_nrms"]
                if first_quant_groups is not None else None
            ),
            "tail_quantization_group_max_nrms": (
                tail_quant_groups["max_group_nrms"]
            ),
        },
        "split_vs_m32_output_nrms": nrms(full, split),
    }
    result["precision_screen"] = {
        "threshold": args.threshold,
        "global_pass": result["split_m1_plus_m31"]["global_nrms"] < args.threshold,
        "row_pass": result["split_m1_plus_m31"]["max_row_nrms"] < args.threshold,
        "first_fixed_32_channel_group_pass": (
            first_eval_groups["max_group_nrms"] < args.threshold
        ),
        "tail_fixed_32_channel_group_pass": (
            tail_eval_groups["max_group_nrms"] < args.threshold
        ),
    }
    result["precision_screen"]["pass"] = all(
        result["precision_screen"][key]
        for key in (
            "global_pass",
            "row_pass",
            "first_fixed_32_channel_group_pass",
            "tail_fixed_32_channel_group_pass",
        )
    )

    rendered = json.dumps(result, indent=2)
    print(rendered)
    output_path = Path(args.output) if args.output else (
        root / (
            f"SPLIT-ACCURACY-{args.backend}-first-g{args.first_group_size}"
            f"-tail-g{args.tail_group_size}.json"
        )
    )
    output_path.write_text(rendered + "\n")
    if not result["precision_screen"]["pass"]:
        raise SystemExit("reassembled split precision screen failed")


if __name__ == "__main__":
    main()
