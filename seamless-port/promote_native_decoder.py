"""Promote the exact validated native decoder without re-encoding weights."""
import argparse
import json
from pathlib import Path
import shutil
import onnx

def promote(directory):
    d=Path(directory)
    stem="decoder.native8-asymmetric-all"
    report=d/(stem+".validated.json")
    model=d/(stem+".onnx")
    check=json.loads(report.read_text())
    if (check.get("component"),check.get("precision"),check.get("policy"),check.get("parity"))!=("decoder","int8-native-experimental","native8-group16-asymmetric-all","PASS"):
        raise ValueError("required native decoder parity gate not passed")
    graph=onnx.load(str(model),load_external_data=False)
    actual=[n for n in graph.graph.node if n.domain=="com.microsoft" and n.op_type=="MatMulNBits"]
    if len(actual)!=48 or check.get("quantized_ops")!=len(actual):
        raise ValueError("expected 48 validated native FFN operators")
    # External weights retain their original name, offsets and contents.
    dest=d/"decoder.int8.onnx"
    shutil.copyfile(model,dest)
    onnx.checker.check_model(str(dest))
    shutil.copyfile(report,d/"decoder.int8.validated.json")
    print("Promoted validated native decoder; no weight conversion performed",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("directory",type=Path)
    promote(p.parse_args().directory)
