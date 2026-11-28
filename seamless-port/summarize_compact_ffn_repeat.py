#!/usr/bin/env python3
"""Summarize balanced Pixel compact-FFN timing repeats and verify every output."""
import argparse
import json
import re
from pathlib import Path
import numpy as np

TIMING = re.compile(
    r"count=(\d+)\s+first=(\d+)\s+curr=(\d+)\s+min=(\d+)\s+max=(\d+)"
    r"\s+avg=([0-9.]+)\s+std=(\d+)\s+p5=(\d+)\s+median=(\d+)\s+p95=(\d+)"
)
INIT = re.compile(
    r"Inference timings in us:\s*Init:\s*(\d+),\s*First inference:\s*(\d+),"
    r"\s*Warmup \(avg\):\s*([0-9.]+),\s*Inference \(avg\):\s*([0-9.]+)"
)
RUN = re.compile(r"repeat-r(\d+)-k(448|512|1024)\.log$")


def nrms(reference, actual):
    return float(np.linalg.norm(actual - reference) /
                 max(float(np.linalg.norm(reference)), 1e-12))


def read_f32(path):
    data = np.fromfile(path, dtype="<f4")
    if data.size != 1024 or not np.isfinite(data).all():
        raise ValueError(f"{path}: expected 1024 finite float32 values")
    return data.astype(np.float64)


def parse_log(path):
    content = path.read_text(errors="replace")
    timings = TIMING.findall(content)
    init = INIT.findall(content)
    if not timings or not init:
        raise ValueError(f"{path}: missing TFLite timing summary")
    t, i = timings[-1], init[-1]
    delegated = ("Replacing 2 out of 2 node(s) with delegate (TfLiteXNNPackDelegate)" in content
                 and "completely executed by the delegate" in content)
    if not delegated:
        raise ValueError(f"{path}: model was not fully delegated to XNNPACK")
    return {
        "count": int(t[0]), "first_us": int(t[1]), "current_us": int(t[2]),
        "min_us": int(t[3]), "max_us": int(t[4]), "avg_us": float(t[5]),
        "std_us": int(t[6]), "p5_us": int(t[7]), "median_us": int(t[8]),
        "p95_us": int(t[9]), "init_us": int(i[0]),
        "first_inference_us": int(i[1]), "warmup_avg_us": float(i[2]),
        "reported_avg_us": float(i[3]), "full_xnnpack": delegated,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path, nargs="?", default=Path("."))
    root = p.parse_args().directory
    reference = read_f32(root / "first-row.source-reference.bin")
    results = []
    for log in sorted(root.glob("repeat-r*-k*.log")):
        match = RUN.search(log.name)
        if not match:
            continue
        round_number, width = map(int, match.groups())
        timing = parse_log(log)
        output_path = root / f"repeat-r{round_number}-k{width}.bin"
        actual = read_f32(output_path)
        host = read_f32(root / f"ffn-m1-compact-k{width}-fp32.host-output.bin")
        groups = [nrms(reference[s:s+32], actual[s:s+32]) for s in range(0, 1024, 32)]
        source_error, host_error = nrms(reference, actual), nrms(host, actual)
        result = {
            "round": round_number, "width": width, "timing": timing,
            "source_nrms": source_error, "max_fixed32_group_nrms": max(groups),
            "host_device_nrms": host_error,
            "accuracy_pass": source_error < 0.05 and max(groups) < 0.05,
            "output_file": output_path.name,
        }
        if not result["accuracy_pass"]:
            raise SystemExit(f"accuracy gate failed: {output_path}: {result}")
        results.append(result)

    rounds = sorted({r["round"] for r in results})
    if not results or len(results) != len(rounds) * 3:
        raise SystemExit(f"expected 3 results per completed round, found {len(results)}")
    by_width = {}
    for width in (448, 512, 1024):
        rows = [r for r in results if r["width"] == width]
        if not rows:
            raise SystemExit(f"no repeat results for width {width}")
        means = np.array([r["timing"]["reported_avg_us"] for r in rows])
        medians = np.array([r["timing"]["median_us"] for r in rows])
        p95s = np.array([r["timing"]["p95_us"] for r in rows])
        maxima = np.array([r["timing"]["max_us"] for r in rows])
        by_width[str(width)] = {
            "runs": len(rows),
            "median_run_average_ms": float(np.median(means) / 1000),
            "p90_run_average_ms": float(np.percentile(means, 90) / 1000),
            "min_run_average_ms": float(np.min(means) / 1000),
            "max_run_average_ms": float(np.max(means) / 1000),
            "median_run_median_ms": float(np.median(medians) / 1000),
            "median_run_p95_ms": float(np.median(p95s) / 1000),
            "worst_run_p95_ms": float(np.max(p95s) / 1000),
            "median_run_max_ms": float(np.median(maxima) / 1000),
            "median_init_ms": float(np.median([r["timing"]["init_us"] for r in rows]) / 1000),
            "all_accuracy_pass": all(r["accuracy_pass"] for r in rows),
            "all_full_xnnpack": all(r["timing"]["full_xnnpack"] for r in rows),
            "per_round_average_ms": {str(r["round"]): r["timing"]["reported_avg_us"] / 1000 for r in rows},
        }
    summary = {
        "device": "Pixel 9a; see per-run device snapshots",
        "method": "balanced schedule; fresh Activity per run; 50 warmups; 1000 measured iterations; 8s rest",
        "rounds": len(rounds), "runs_per_width": min(v["runs"] for v in by_width.values()),
        "accuracy_threshold": 0.05,
        "all_outputs_pass": all(r["accuracy_pass"] for r in results),
        "all_runs_full_xnnpack": all(r["timing"]["full_xnnpack"] for r in results),
        "by_width": by_width, "runs": results,
    }
    rendered = json.dumps(summary, indent=2)
    (root / "COMPACT-REPEATABILITY.json").write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
