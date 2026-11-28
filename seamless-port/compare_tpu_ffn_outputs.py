#!/usr/bin/env python3
"""Reconstruct and strictly audit grouped INT8 FFN outputs from Android TFLite."""
import argparse
import json
from pathlib import Path

import numpy as np


def nrms(reference, actual):
    reference = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    return float(
        np.linalg.norm(actual - reference)
        / max(float(np.linalg.norm(reference)), 1e-12)
    )


def row_error_metrics(reference, actual):
    reference = np.asarray(reference)
    actual = np.asarray(actual)
    if reference.shape != actual.shape or reference.ndim != 2:
        raise ValueError("row metrics require equal two-dimensional tensors")
    errors = [nrms(reference[i:i + 1], actual[i:i + 1])
              for i in range(reference.shape[0])]
    worst = int(np.argmax(errors))
    return {
        "row_nrms": errors,
        "p50_row_nrms": float(np.percentile(errors, 50)),
        "p95_row_nrms": float(np.percentile(errors, 95)),
        "max_row_nrms": float(errors[worst]),
        "worst_row_index": worst,
    }


def group_error_metrics(reference, actual, group_size):
    reference = np.asarray(reference)
    actual = np.asarray(actual)
    if reference.shape != actual.shape or reference.ndim != 2:
        raise ValueError("group metrics require equal two-dimensional tensors")
    if group_size <= 0 or reference.shape[1] % group_size:
        raise ValueError("group size must divide the output channel count")
    errors = [
        nrms(reference[:, start:start + group_size],
             actual[:, start:start + group_size])
        for start in range(0, reference.shape[1], group_size)
    ]
    worst = int(np.argmax(errors))
    return {
        "group_nrms": errors,
        "p50_group_nrms": float(np.percentile(errors, 50)),
        "p95_group_nrms": float(np.percentile(errors, 95)),
        "max_group_nrms": float(errors[worst]),
        "worst_group_index": worst,
        "worst_group_channels": [
            worst * group_size, (worst + 1) * group_size
        ],
        "groups_over_5pct": int(sum(error >= 0.05 for error in errors)),
    }


def split_prefill_rows(x, y):
    """Split the exceptional first prefill row from the remaining rows."""
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim != 2 or y.shape != x.shape or x.shape[0] < 2:
        raise ValueError("prefill split requires matching [rows, channels] arrays")
    return (x[:1], y[:1]), (x[1:], y[1:])


def reconstruct(raw, meta):
    rows = int(meta["rows"])
    group_size = int(meta["group_size"])
    count = int(meta["output_groups"])
    q = np.fromfile(raw, dtype=np.int8)
    expected = count * rows * group_size
    if q.size != expected:
        raise ValueError(f"{raw}: {q.size} bytes, expected {expected}")
    parts = q.reshape(count, rows, group_size)
    actual = np.empty((rows, count * group_size), dtype=np.float32)
    mapping = meta["output_group_to_tensor_index"]
    scales = meta["output_quantizations"]
    if len(mapping) != count or len(scales) != count:
        raise ValueError("manifest output inventory mismatch")
    if sorted(mapping) != list(range(count)):
        raise ValueError("manifest output mapping is not a permutation")
    for channel_group, tensor_index in enumerate(mapping):
        scale, zero = scales[tensor_index]
        if float(scale) <= 0:
            raise ValueError("invalid output quantization scale")
        actual[:, channel_group * group_size:(channel_group + 1) * group_size] = (
            (parts[tensor_index].astype(np.float32) - int(zero)) * float(scale)
        )
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("variant")
    parser.add_argument("device_output")
    parser.add_argument("--host-output", default=None)
    parser.add_argument("--global-threshold", type=float, default=0.05)
    parser.add_argument("--row-threshold", type=float, default=0.05)
    parser.add_argument("--group-threshold", type=float, default=0.05)
    parser.add_argument("--require-exact-backend-parity", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    rows = json.loads(manifest_path.read_text())
    meta = next((item for item in rows if item["variant"] == args.variant), None)
    if meta is None:
        raise ValueError("variant not present in manifest")
    if meta["status"] != "CONVERTED":
        raise ValueError("variant was not converted")

    reference = np.fromfile(
        manifest_path.parent / (args.variant + ".source-reference.bin"),
        dtype="<f4",
    ).reshape(int(meta["rows"]), 1024)
    device = reconstruct(args.device_output, meta)
    device_global = nrms(reference, device)
    device_rows = row_error_metrics(reference, device)
    device_groups = group_error_metrics(
        reference, device, int(meta["group_size"])
    )
    result = {
        "variant": args.variant,
        "workload": meta.get("workload"),
        "device_output": args.device_output,
        "source_row_offset": int(meta.get("source_row_offset", 0)),
        "device_vs_fp32_nrms": device_global,
        "device_vs_fp32_max_abs": float(np.max(np.abs(device - reference))),
        "device_vs_fp32_rows": device_rows,
        "device_vs_fp32_groups": device_groups,
    }
    result["device_vs_fp32_rows"]["worst_source_row_index"] = (
        result["source_row_offset"] + device_rows["worst_row_index"]
    )

    precision_pass = (
        device_global < args.global_threshold
        and device_rows["max_row_nrms"] < args.row_threshold
        and device_groups["max_group_nrms"] < args.group_threshold
    )
    result["precision_screen"] = {
        "global_pass": device_global < args.global_threshold,
        "row_pass": device_rows["max_row_nrms"] < args.row_threshold,
        "group_pass": device_groups["max_group_nrms"] < args.group_threshold,
        "pass": bool(precision_pass),
        "thresholds": {
            "global_nrms": args.global_threshold,
            "row_nrms": args.row_threshold,
            "group_nrms": args.group_threshold,
        },
    }

    host_path = Path(args.host_output) if args.host_output else (
        manifest_path.parent / (args.variant + ".cpu-output.bin")
    )
    failures = []
    if not precision_pass:
        failures.append("global, per-row, or per-group precision screen failed")
    if host_path.exists():
        host = reconstruct(host_path, meta)
        host_rows = row_error_metrics(reference, host)
        host_groups = group_error_metrics(
            reference, host, int(meta["group_size"])
        )
        device_raw = Path(args.device_output).read_bytes()
        host_raw = host_path.read_bytes()
        byte_differences = sum(a != b for a, b in zip(device_raw, host_raw))
        byte_differences += abs(len(device_raw) - len(host_raw))
        exact = device_raw == host_raw
        result.update({
            "host_cpu_vs_fp32_nrms": nrms(reference, host),
            "host_cpu_vs_fp32_rows": host_rows,
            "host_cpu_vs_fp32_groups": host_groups,
            "device_vs_host_cpu_nrms": nrms(host, device),
            "device_vs_host_cpu_max_abs": float(np.max(np.abs(device - host))),
            "device_vs_host_cpu_exact_bytes": exact,
            "device_vs_host_cpu_different_bytes": int(byte_differences),
        })
        if args.require_exact_backend_parity and not exact:
            failures.append("device and host CPU output bytes differ")
    elif args.require_exact_backend_parity:
        failures.append("host CPU output file is missing")

    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
