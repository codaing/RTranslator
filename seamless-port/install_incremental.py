"""Replace the legacy decoder in a verified extracted package with the validated FP16 candidate."""
import json
from pathlib import Path
import re
import shutil
import sys
import onnx
from model_pack import tensors,digest

def refs(graph):
    model=onnx.load(str(graph),load_external_data=False)
    names=set()
    for t in tensors(model.graph):
        for e in t.external_data:
            if e.key=="location":
                if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,240}",e.value) or e.value in (".",".."):
                    raise ValueError("unsafe external data reference")
                names.add(e.value)
    return names

def main(base,candidate):
    base,candidate=Path(base),Path(candidate)
    manifest=json.loads((base/"sha256.json").read_text())
    actual={p.name for p in base.iterdir() if p.is_file() and p.name!="sha256.json"}
    if actual!=set(manifest):raise ValueError("base manifest inventory mismatch")
    for name in sorted(actual):
        if digest(base/name)!=manifest[name]:raise ValueError("base hash mismatch: "+name)
    old=base/"decoder.int8.onnx"
    name="decoder.incremental.fp16.onnx"
    graph=candidate/name
    marker=candidate/"decoder.incremental.fp16.validated.json"
    meta=candidate/"decoder.incremental.meta.json"
    report=json.loads(marker.read_text())
    if report.get("parity")!="PASS" or report.get("precision")!="fp16-incremental-experimental" or report.get("component")!="decoder":
        raise ValueError("FP16 incremental parity gate missing")
    if not report.get("tests"):raise ValueError("empty incremental validation")
    onnx.checker.check_model(str(graph))
    kept=set()
    for n in ("encoder.int8.onnx","units.onnx","vocoder.onnx"):kept.update(refs(base/n))
    remove=refs(old)-kept
    for n in remove:(base/n).unlink()
    old.unlink()
    for n in ("decoder.int8.validated.json","speech-smoke.json","sha256.json"):(base/n).unlink(missing_ok=True)
    for src in [graph,marker,meta]+[candidate/n for n in sorted(refs(graph))]:
        if (base/src.name).exists():raise ValueError("candidate filename collision "+src.name)
        shutil.move(str(src),str(base/src.name))
    (base/"MODEL-LICENSE.txt").write_text((base/"MODEL-LICENSE.txt").read_text()+"\nAdditional modification: FP16 incremental translation decoder with explicit self-attention KV cache.\n")
    print("Installed validated incremental candidate; package bytes",sum(p.stat().st_size for p in base.iterdir()),flush=True)

if __name__=="__main__":main(*sys.argv[1:])
