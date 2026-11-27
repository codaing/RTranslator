"""Stage only validated model graphs + referenced external data; create flat import ZIP."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import re
import shutil
import zipfile
import onnx
from onnx.external_data_helper import uses_external_data

NAMES={"encoder":"encoder.int8.onnx","decoder":"decoder.int8.onnx",
       "units":"units.onnx","vocoder":"vocoder.onnx"}
NOTICE="""SeamlessStreaming model weights: Meta Platforms, Inc.
Converted from facebookresearch/seamless_communication, revision 9a081e935c29c6b1e0e05bcbf3fcb623ad306524.
Model weights licensed CC BY-NC 4.0: https://creativecommons.org/licenses/by-nc/4.0/
Source: https://github.com/facebookresearch/seamless_communication
Modifications: ONNX export, dynamic graph repair, INT8 weight quantization of encoder and native asymmetric grouped INT8 decoder feed-forward projections with floating-point inputs.
Noncommercial use only. Experimental conversion; no warranty.
"""


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def copy_unique(src,dst):
    """Staging sources are immutable; hardlink where possible to save disk."""
    src,dst=Path(src),Path(dst)
    if not src.is_file():raise FileNotFoundError(src)
    start=time.monotonic()
    size=src.stat().st_size
    if dst.exists():
        if not src.samefile(dst):
            if size!=dst.stat().st_size or digest(src)!=digest(dst):
                raise ValueError(f"conflicting filename: {dst.name}")
        method="reuse"
    else:
        try:
            os.link(src,dst)
            method="hardlink"
        except OSError:
            if shutil.disk_usage(dst.parent).free<size:
                raise OSError(f"insufficient space for {dst.name}: {size} bytes")
            shutil.copyfile(src,dst)
            method="copy"
    print(f"{method} {dst.name}: {size:,} bytes ({time.monotonic()-start:.2f}s)",flush=True)


def tensors(graph):
    yield from graph.initializer
    for node in graph.node:
        for attr in node.attribute:
            if attr.type==onnx.AttributeProto.TENSOR:
                yield attr.t
            elif attr.type==onnx.AttributeProto.TENSORS:
                yield from attr.tensors
            elif attr.type==onnx.AttributeProto.GRAPH:
                yield from tensors(attr.g)
            elif attr.type==onnx.AttributeProto.GRAPHS:
                for g in attr.graphs:
                    yield from tensors(g)


def stage(component,source,destination):
    source,destination=Path(source),Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    name=NAMES[component]
    marker=source/(name.removesuffix(".onnx")+".validated.json")
    report=json.loads(marker.read_text())
    expected={"encoder":"int8-dynamic","decoder":"int8-native-experimental"}.get(component,"fp32")
    if report.get("component")!=component or report.get("parity")!="PASS" or report.get("precision")!=expected:
        raise ValueError(f"missing validated {component} {expected}")
    if component=="decoder" and report.get("policy")!="native8-group16-asymmetric-all":
        raise ValueError("decoder must use the validated asymmetric native policy")
    graph=onnx.load(str(source/name),load_external_data=False)
    if component=="decoder" and not any(n.op_type=="MatMulNBits" and n.domain=="com.microsoft" for n in graph.graph.node):
        raise ValueError("decoder is missing native packed matrix multiplication")
    copy_unique(source/name,destination/name)
    locations=set()
    refs=0
    for t in tensors(graph.graph):
        if uses_external_data(t):
            info={e.key:e.value for e in t.external_data}
            location=info["location"]
            if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,240}",location) or location in (".",".."):
                raise ValueError(f"unsafe or nested external file: {location}")
            locations.add(location)
            refs+=1
    print(f"Stage {component}: {refs:,} external tensor references, {len(locations):,} unique files",flush=True)
    for location in sorted(locations):
        copy_unique(source/location,destination/location)
    copy_unique(marker,destination/marker.name)
    if component=="units":
        copy_unique(source/"tokenizer.json",destination/"tokenizer.json")


def merge(source,destination):
    destination.mkdir(parents=True,exist_ok=True)
    for component in NAMES:
        stage(component,source/("model-part-"+component),destination)
    (destination/"MODEL-LICENSE.txt").write_text(NOTICE)


def pack(source,output):
    required=set(NAMES.values())|{"tokenizer.json","MODEL-LICENSE.txt"}
    files=sorted(p for p in source.iterdir() if p.is_file() and p.name!="sha256.json")
    if output.resolve().parent==source.resolve():
        raise ValueError("output ZIP must be outside model folder")
    if not required.issubset({p.name for p in files}):
        raise ValueError("missing required model files")
    # This ZIP is generated only after the four-model smoke test succeeds.
    check=json.loads((source/"speech-smoke.json").read_text())
    if check.get("result")!="PASS":
        raise ValueError("speech smoke gate not passed")
    total=sum(p.stat().st_size for p in files)
    if shutil.disk_usage(output.parent).free<total+8*1024*1024:
        raise OSError(f"insufficient free disk for {total:,} byte model ZIP")
    hashes={}
    for p in files:
        print(f"Hashing {p.name} ({p.stat().st_size:,} bytes)",flush=True)
        hashes[p.name]=digest(p)
    (source/"sha256.json").write_text(json.dumps(hashes,indent=2)+"\n")
    with zipfile.ZipFile(output,"w",compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        for p in sorted(source.iterdir()):
            if p.is_file():
                print(f"Adding {p.name} ({p.stat().st_size:,} bytes)",flush=True)
                z.write(p,p.name)
    print(f"Model ZIP: {output.stat().st_size:,} bytes",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    subs=p.add_subparsers(dest="command",required=True)
    s=subs.add_parser("stage");s.add_argument("component",choices=NAMES);s.add_argument("source",type=Path);s.add_argument("destination",type=Path)
    for cmd in ("merge","pack"):
        s=subs.add_parser(cmd);s.add_argument("source",type=Path);s.add_argument("destination",type=Path)
    a=p.parse_args()
    if a.command=="stage":stage(a.component,a.source,a.destination)
    elif a.command=="merge":merge(a.source,a.destination)
    else:pack(a.source,a.destination)
