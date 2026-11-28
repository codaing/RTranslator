#!/usr/bin/env python3
"""Validate compact physical outputs against original FP16 and host TFLite."""
import argparse
import json
from pathlib import Path
import re
import numpy as np

def metrics(y, z):
    if y.shape != z.shape or not np.isfinite(z).all():
        raise ValueError("invalid output")
    def ratio(a, b):
        return float(np.linalg.norm(a-b) / max(np.linalg.norm(a), 1e-12))
    return {"nrms": ratio(y, z),
            "max_fixed32_nrms": max(ratio(a,b) for a,b in zip(y.reshape(32,32),z.reshape(32,32))),
            "max_abs": float(np.max(np.abs(y-z)))}

def read(path):
    if path.stat().st_size != 4096:
        raise ValueError(f"{path}: expected exactly 4096 bytes")
    return np.fromfile(path, dtype="<f4").astype(np.float64)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path, nargs="?", default=Path("."))
    a = p.parse_args()
    root = a.directory
    meta = json.loads((root / "compact-manifest.json").read_text())
    ref = read(root / "first-row.source-reference.bin")
    results = []
    for v in meta["variants"]:
        name = v["variant"]
        device = read(root / (name+"-cpu-output.bin"))
        host = read(root / (name+".host-output.bin"))
        log = (root / (name+"-cpu.log")).read_text()
        timing = re.findall(r"Inference \(avg\): ([0-9.]+)", log)
        delegated = bool(re.search(r"Replacing 2 out of 2 node\(s\) with delegate \(TfLiteXNNPackDelegate\)", log))
        if not timing or not delegated:
            raise ValueError(f"{name}: missing benchmark timing or complete XNNPACK delegation")
        source = metrics(ref, device)
        result = {"variant": name, "source": source, "host": metrics(host, device),
                  "average_ms": float(timing[-1])/1000,
                  "full_xnnpack": delegated,
                  "precision_pass": source["nrms"] < .05 and source["max_fixed32_nrms"] < .05}
        results.append(result)
    print(json.dumps(results, indent=2))
    if not all(v["precision_pass"] for v in results):
        raise SystemExit("compact physical precision screen failed")

if __name__ == "__main__":
    main()
