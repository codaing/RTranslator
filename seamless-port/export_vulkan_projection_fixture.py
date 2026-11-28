#!/usr/bin/env python3
"""Export one real Seamless decoder source-projection MatMul for on-device Vulkan parity."""
import argparse
import copy
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper as nh
from onnx.external_data_helper import load_external_data_for_tensor
import onnxruntime as ort

MAGIC = b"SVP1"
HEADER = struct.Struct("<4sIIII")  # magic, version, M, K, N


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def pack_fixture(activation, weight, reference):
    """Pack X[M,K] FP32, W[N,K] FP16 output-major, with a 20-byte header."""
    x = np.ascontiguousarray(activation, dtype="<f4")
    w = np.ascontiguousarray(weight.T, dtype="<f2")
    y = np.ascontiguousarray(reference, dtype="<f4")
    if x.ndim != 2 or w.ndim != 2 or y.ndim != 2:
        raise ValueError("fixture arrays must be matrices")
    m, k = x.shape
    n, wk = w.shape
    if wk != k or y.shape != (m, n) or k % 2 or n % 4:
        raise ValueError("incompatible projection dimensions")
    return HEADER.pack(MAGIC, 1, m, k, n) + x.tobytes() + w.tobytes()


def parse_fixture_header(data):
    if len(data) < HEADER.size:
        raise ValueError("truncated projection fixture")
    magic, version, m, k, n = HEADER.unpack_from(data)
    if magic != MAGIC or version != 1:
        raise ValueError("unsupported projection fixture")
    expected = HEADER.size + m * k * 4 + n * k * 2
    if len(data) != expected:
        raise ValueError(f"fixture size mismatch: {len(data)} != {expected}")
    return {"M": m, "K": k, "N": n, "bytes": expected}


def _attr(node, name):
    for a in node.attribute:
        if a.name == name:
            return a
    return None


def _cast_dtype(elem_type):
    return {
        T.FLOAT: np.float32, T.FLOAT16: np.float16, T.DOUBLE: np.float64,
        T.INT8: np.int8, T.UINT8: np.uint8,
    }.get(elem_type)


def _constant_info(name, initializers, producers, seen=None):
    seen = set() if seen is None else seen
    if name in seen:
        return None
    seen.add(name)
    tensor = initializers.get(name)
    if tensor is not None:
        return tuple(tensor.dims), tensor.data_type, ("initializer", tensor.name)
    node = producers.get(name)
    if node is None or not node.input:
        return None
    if node.op_type in ("Identity", "Transpose", "Cast"):
        info = _constant_info(node.input[0], initializers, producers, seen)
        if info is None:
            return None
        shape, dtype, source = info
        if node.op_type == "Transpose":
            attr = _attr(node, "perm")
            perm = list(attr.ints) if attr is not None else list(reversed(range(len(shape))))
            shape = tuple(shape[i] for i in perm)
        elif node.op_type == "Cast":
            attr = _attr(node, "to")
            if attr is None:
                return None
            dtype = attr.i
        return shape, dtype, source
    return None


def _constant_array(name, initializers, producers, base_dir, cache=None):
    cache = {} if cache is None else cache
    if name in cache:
        return cache[name]
    tensor = initializers.get(name)
    if tensor is not None:
        if tensor.data_location == T.EXTERNAL:
            load_external_data_for_tensor(tensor, str(base_dir))
        value = nh.to_array(tensor)
    else:
        node = producers.get(name)
        if node is None or not node.input:
            return None
        value = _constant_array(node.input[0], initializers, producers, base_dir, cache)
        if value is None:
            return None
        if node.op_type == "Identity":
            pass
        elif node.op_type == "Transpose":
            attr = _attr(node, "perm")
            perm = list(attr.ints) if attr is not None else list(reversed(range(value.ndim)))
            value = value.transpose(perm)
        elif node.op_type == "Cast":
            attr = _attr(node, "to")
            dtype = _cast_dtype(attr.i) if attr is not None else None
            if dtype is None:
                return None
            value = value.astype(dtype)
        else:
            return None
    cache[name] = value
    return value


def _value_info_map(model):
    result = {}
    for value in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        result[value.name] = value
    return result


def _nrms(actual, reference):
    a = np.asarray(actual, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    return float(np.linalg.norm(a - b) / max(float(np.linalg.norm(b)), 1e-12))


def _candidate_nodes(model):
    initializers = {x.name: x for x in model.graph.initializer}
    producers = {o: n for n in model.graph.node for o in n.output if o}
    candidates = []
    for node in model.graph.node:
        if node.op_type != "MatMul" or len(node.input) != 2 or not node.output:
            continue
        activation_name, weight_name = node.input
        info = _constant_info(weight_name, initializers, producers)
        if info is None:
            continue
        shape, dtype, source = info
        if len(shape) != 2 or dtype != T.FLOAT16:
            continue
        k, n = shape
        if k < 32 or n < 4 or k % 2 or n % 4 or k > 8192 or n > 8192:
            continue
        if activation_name in initializers:
            continue
        candidates.append((k * n, node, activation_name, weight_name, source, k, n))
    candidates.sort(key=lambda x: (x[0], x[5], x[6]), reverse=True)
    return candidates, initializers, producers


def _capture_value_info(model, names):
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False, data_prop=False)
    values = _value_info_map(inferred)
    missing = [name for name in names if name not in values]
    if missing:
        raise ValueError("shape inference missing capture tensors: " + ", ".join(missing))
    capture = copy.deepcopy(inferred)
    existing = {x.name for x in capture.graph.output}
    for name in names:
        if name not in existing:
            capture.graph.output.append(copy.deepcopy(values[name]))
            existing.add(name)
    return capture


def export(directory, fbank_path, output_dir):
    directory = Path(directory)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder_path = directory / "encoder.int8.onnx"
    source_path = directory / "decoder.source-projections.onnx"
    if not encoder_path.is_file() or not source_path.is_file():
        raise FileNotFoundError("validated encoder.int8.onnx and decoder.source-projections.onnx are required")

    features = np.fromfile(fbank_path, dtype="<f4")
    if features.size % 80:
        raise ValueError("filterbank file is not a multiple of 80 float32 values")
    features = features.reshape(1, -1, 80)
    if features.shape[1] < 192:
        raise ValueError("real speech fixture is too short")

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    encoder = ort.InferenceSession(str(encoder_path), options, providers=["CPUExecutionProvider"])
    encoder_input = encoder.get_inputs()[0].name
    encoded_name = next((x.name for x in encoder.get_outputs() if x.name == "encoded"), None)
    if encoded_name is None:
        raise ValueError("encoder graph has no encoded output")
    encoded = encoder.run([encoded_name], {encoder_input: features})[0]
    del encoder

    source_model = onnx.load(str(source_path), load_external_data=False)
    candidates, initializers, producers = _candidate_nodes(source_model)
    if not candidates:
        raise RuntimeError("no compatible FP16 source-projection MatMul found")
    failures = []
    for _, node, activation_name, weight_name, weight_source, k, n in candidates:
        try:
            weight = _constant_array(weight_name, initializers, producers, directory)
            if weight is None or weight.shape != (k, n) or weight.dtype != np.float16:
                raise ValueError("resolved weight is not the expected FP16 [K,N] matrix")
            capture_model = _capture_value_info(source_model, [activation_name, node.output[0]])
            capture_path = directory / "decoder.source-projections.capture.onnx"
            onnx.save_model(capture_model, str(capture_path))
            capture_options = ort.SessionOptions()
            capture_options.intra_op_num_threads = 4
            capture_options.inter_op_num_threads = 1
            capture = ort.InferenceSession(str(capture_path), capture_options,
                                           providers=["CPUExecutionProvider"])
            encoded_input = capture.get_inputs()[0].name
            activation, original_output = capture.run(
                [activation_name, node.output[0]], {encoded_input: encoded})
            del capture

            if activation.ndim < 2 or activation.shape[-1] != k:
                raise ValueError(f"activation shape {activation.shape} does not end in K={k}")
            x = np.ascontiguousarray(activation.reshape(-1, k), dtype=np.float32)
            m = x.shape[0]
            if original_output.size != m * n or m < 1 or m > 2048:
                raise ValueError(f"MatMul output shape {original_output.shape} is incompatible with M={m}, N={n}")
            if not np.isfinite(x).all() or not np.isfinite(weight).all():
                raise ValueError("non-finite model activation or weight")

            reference_model = h.make_model(
                h.make_graph(
                    [h.make_node("MatMul", ["x", "weight"], ["y"], name="real_model_projection")],
                    "real_model_projection_reference",
                    [h.make_tensor_value_info("x", T.FLOAT, [m, k])],
                    [h.make_tensor_value_info("y", T.FLOAT, [m, n])],
                    [nh.from_array(np.ascontiguousarray(weight, dtype=np.float32), name="weight")],
                ),
                opset_imports=[h.make_opsetid("", 17)], ir_version=9,
            )
            onnx.checker.check_model(reference_model)
            reference_path = output_dir / "projection-reference.onnx"
            onnx.save(reference_model, str(reference_path))
            reference_options = ort.SessionOptions()
            reference_options.intra_op_num_threads = 4
            reference_options.inter_op_num_threads = 1
            reference_session = ort.InferenceSession(str(reference_path), reference_options,
                                                     providers=["CPUExecutionProvider"])
            reference = reference_session.run(["y"], {"x": x})[0]
            del reference_session
            reference = np.ascontiguousarray(reference.reshape(m, n), dtype=np.float32)
            original = np.asarray(original_output, dtype=np.float32).reshape(m, n)
            source_fp16_nrms = _nrms(reference, original)

            blob = pack_fixture(x, weight, reference)
            header = parse_fixture_header(blob)
            fixture_path = output_dir / "vulkan-real-projection.bin"
            fixture_path.write_bytes(blob)
            meta = {
                "source": "Meta SeamlessStreaming incremental FP16 decoder source-projection graph",
                "source_model": source_path.name,
                "source_model_sha256": sha256(source_path.read_bytes()),
                "input_audio": Path(fbank_path).name,
                "input_fbank_sha256": sha256(Path(fbank_path).read_bytes()),
                "encoder": encoder_path.name,
                "encoder_output": encoded_name,
                "node": node.name or node.output[0],
                "node_output": node.output[0],
                "activation": activation_name,
                "weight_input": weight_name,
                "weight_initializer_source": weight_source,
                "activation_dtype": str(activation.dtype),
                "weight_dtype": str(weight.dtype),
                "source_node_output_dtype": str(original_output.dtype),
                "M": m, "K": k, "N": n,
                "layout": "X[M,K] FP32; W[N,K] FP16 output-major; Y[M,N] FP32",
                "reference": "isolated ONNX Runtime CPUExecutionProvider FP32 MatMul using exact FP16 model weights promoted to FP32",
                "ort_version": ort.__version__,
                "source_fp16_output_vs_fp32_reference_nrms": source_fp16_nrms,
                "fixture_sha256": sha256(blob),
                "reference_model_sha256": sha256(reference_path.read_bytes()),
                "fixture_bytes": header["bytes"],
                "license": "Meta SeamlessStreaming CC BY-NC 4.0; noncommercial use only",
            }
            (output_dir / "vulkan-real-projection.json").write_text(
                json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(meta, indent=2), flush=True)
            return
        except Exception as exc:
            failures.append(f"{node.name or node.output[0]} K={k} N={n}: {exc}")
            print("candidate rejected: " + failures[-1], flush=True)
    raise RuntimeError("all candidate MatMul exports failed:\n" + "\n".join(failures))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model_directory", type=Path)
    p.add_argument("fbank", type=Path)
    p.add_argument("output_directory", type=Path)
    a = p.parse_args()
    export(a.model_directory, a.fbank, a.output_directory)


if __name__ == "__main__":
    main()
