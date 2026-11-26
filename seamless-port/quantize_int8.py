#!/usr/bin/env python3
"""Experimental weight INT8 quantization of validated SeamlessStreaming graphs."""
import argparse
import json
from pathlib import Path
import time


def quantize(component, directory):
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    directory = Path(directory)
    source = directory / f"{component}.onnx"
    report = directory / f"{component}.validated.json"
    output = directory / f"{component}.int8.onnx"
    marker = directory / f"{component}.int8.validated.json"
    marker.unlink(missing_ok=True)
    baseline = json.loads(report.read_text())
    if baseline.get("parity") != "PASS" or baseline.get("precision") != "fp32" or baseline.get("component") != component:
        raise ValueError("A validated FP32 graph is required")
    cases = sorted(directory.glob(f"{component}.case*.npz"))
    if not cases:
        raise ValueError("No saved FP32 validation fixtures")
    if len(cases) != len(baseline.get("tests", [])):
        raise ValueError("Fixture count differs from FP32 validation report")
    # Do not quantize arbitrary activation-only matrix multiplication.
    quantize_dynamic(
        str(source), str(output), weight_type=QuantType.QInt8,
        per_channel=True, op_types_to_quantize=["MatMul", "Gemm"],
        extra_options={"MatMulConstBOnly": True}, use_external_data_format=True,
    )
    onnx.checker.check_model(str(output))
    graph = onnx.load(str(output), load_external_data=False)
    quantized_ops = sum(n.op_type in ("MatMulInteger", "QLinearMatMul", "DynamicQuantizeMatMul", "QGemm")
                        for n in graph.graph.node)
    if not quantized_ops:
        raise RuntimeError("No INT8 matrix multiplications were produced")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(output), sess_options=options, providers=["CPUExecutionProvider"])
    results = []
    for case in cases:
        with np.load(case, allow_pickle=False) as fixture:
            inputs = {i.name: fixture["input_" + i.name] for i in session.get_inputs()}
            names = [o.name for o in session.get_outputs()]
            session.run(names, inputs)  # warm-up
            start = time.monotonic()
            values = session.run(names, inputs)
            elapsed = time.monotonic() - start
            metrics = {}
            for name, actual in zip(names, values):
                expected = fixture["output_" + name]
                if expected.shape != actual.shape or not np.isfinite(actual).all():
                    raise RuntimeError(f"{case.name}: invalid {name} output")
                delta = actual.astype(np.float64) - expected.astype(np.float64)
                rms = float(np.sqrt(np.mean(delta ** 2)))
                scale = max(float(np.sqrt(np.mean(expected.astype(np.float64) ** 2))), 1e-6)
                nrms = rms / scale
                if nrms > 0.10:
                    raise RuntimeError(f"{case.name}: {name} normalized RMS error {nrms:.3f} exceeds 0.10")
                metrics[name] = {"normalized_rms_error": nrms}
                if name == "logits":
                    agreement = float(np.mean(np.argmax(actual, axis=-1) == np.argmax(expected, axis=-1)))
                    metrics[name]["top_token_agreement"] = agreement
                    if agreement < 1.0:
                        raise RuntimeError(f"{case.name}: top token differs")
                if name == "p_choose":
                    # Reference decision_method=min over all layers/heads at the last target/source position.
                    decision = bool(np.min(actual[..., -1, -1]) >= 0.5)
                    reference = bool(np.min(expected[..., -1, -1]) >= 0.5)
                    metrics[name]["read_write_agreement"] = decision == reference
                    if decision != reference:
                        raise RuntimeError(f"{case.name}: monotonic read/write decision differs")
            results.append({"fixture": case.name, "seconds": elapsed, "metrics": metrics})
    marker.write_text(json.dumps({"component": component, "precision": "int8-dynamic",
                                 "parity": "PASS", "quantized_ops": quantized_ops,
                                 "tests": results}, indent=2) + "\n")
    return marker


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("component", choices=("encoder", "decoder"))
    p.add_argument("directory", type=Path)
    args = p.parse_args()
    print(quantize(args.component, args.directory))


if __name__ == "__main__":
    main()
