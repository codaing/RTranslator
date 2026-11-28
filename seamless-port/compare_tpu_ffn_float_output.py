#!/usr/bin/env python3
"""Compare an Android TFLite FP32 first-row output with the source FP16 core."""
import argparse
import json
from pathlib import Path

import numpy as np

from compare_tpu_ffn_outputs import group_error_metrics, nrms, row_error_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("variant")
    parser.add_argument("device_output")
    parser.add_argument("--host-output", default=None)
    parser.add_argument("--threshold", type=float, default=0.05)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    meta = next(
        (item for item in manifest if item["variant"] == args.variant), None
    )
    if meta is None or meta.get("status") != "CONVERTED":
        raise ValueError("variant is missing or not converted")

    rows = int(meta["rows"])
    reference = np.fromfile(
        root / (args.variant + ".source-reference.bin"), dtype="<f4"
    ).reshape(rows, 1024)

    def read_output(path):
        raw = np.fromfile(path, dtype="<f4")
        if raw.size != rows * 1024:
            raise ValueError(
                f"{path}: {raw.size} float32 values, expected {rows * 1024}"
            )
        return raw.reshape(rows, 1024)

    device = read_output(args.device_output)
    device_rows = row_error_metrics(reference, device)
    device_groups = group_error_metrics(reference, device, 32)
    result = {
        "variant": args.variant,
        "device_output": args.device_output,
        "device_vs_source_fp16_nrms": nrms(reference, device),
        "device_vs_source_fp16_max_abs": float(np.max(np.abs(device - reference))),
        "device_vs_source_fp16_rows": device_rows,
        "device_vs_source_fp16_fixed_32_channel_groups": device_groups,
    }

    host_path = (
        Path(args.host_output) if args.host_output
        else root / (args.variant + ".fp32.cpu-output.bin")
    )
    if not host_path.is_absolute():
        host_path = root / host_path
    if host_path.exists():
        host = read_output(host_path)
        result["host_cpu_vs_source_fp16_nrms"] = nrms(reference, host)
        result["device_vs_host_cpu_nrms"] = nrms(host, device)

    result["precision_screen"] = {
        "threshold": args.threshold,
        "global_pass": result["device_vs_source_fp16_nrms"] < args.threshold,
        "row_pass": device_rows["max_row_nrms"] < args.threshold,
        "fixed_32_channel_group_pass": (
            device_groups["max_group_nrms"] < args.threshold
        ),
    }
    result["precision_screen"]["pass"] = all(
        result["precision_screen"].values()
    )
    print(json.dumps(result, indent=2))
    if not result["precision_screen"]["pass"]:
        raise SystemExit("FP32 first-row precision screen failed")


if __name__ == "__main__":
    main()
