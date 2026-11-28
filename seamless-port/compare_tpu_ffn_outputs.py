#!/usr/bin/env python3
"""Dequantize grouped FFN outputs dumped by the Android TFLite benchmark app."""
import argparse
import json
from pathlib import Path
import numpy as np

def nrms(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    return float(np.linalg.norm(a-b)/max(float(np.linalg.norm(b)),1e-12))

def reconstruct(raw,meta):
    m=int(meta["rows"]);g=int(meta["group_size"]);count=int(meta["output_groups"])
    q=np.fromfile(raw,dtype=np.int8)
    expected=count*m*g
    if q.size!=expected:raise ValueError(f"{raw}: {q.size} bytes, expected {expected}")
    parts=q.reshape(count,m,g)
    actual=np.empty((m,count*g),dtype=np.float32)
    mapping=meta["output_group_to_tensor_index"]
    scales=meta["output_quantizations"]
    if len(mapping)!=count or len(scales)!=count:raise ValueError("manifest output inventory mismatch")
    for channel_group,tensor_index in enumerate(mapping):
        scale,zero=scales[tensor_index]
        actual[:,channel_group*g:(channel_group+1)*g]=(parts[tensor_index].astype(np.float32)-zero)*scale
    return actual

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest");p.add_argument("variant");p.add_argument("device_output")
    p.add_argument("--host-output",default=None)
    a=p.parse_args()
    rows=json.loads(Path(a.manifest).read_text())
    meta=next((x for x in rows if x["variant"]==a.variant),None)
    if meta is None:raise ValueError("variant not present in manifest")
    if meta["status"]!="CONVERTED":raise ValueError("variant was not converted")
    ref=np.fromfile(Path(a.manifest).parent/(a.variant+".source-reference.bin"),dtype="<f4").reshape(meta["rows"],1024)
    device=reconstruct(a.device_output,meta)
    result={"variant":a.variant,"device_output":a.device_output,
        "device_vs_fp32_nrms":nrms(device,ref),"device_vs_fp32_max_abs":float(np.max(np.abs(device-ref)))}
    host_path=a.host_output or str(Path(a.manifest).parent/(a.variant+".cpu-output.bin"))
    if Path(host_path).exists():
        host=reconstruct(host_path,meta)
        result["host_cpu_vs_fp32_nrms"]=nrms(host,ref)
        result["device_vs_host_cpu_nrms"]=nrms(device,host)
        result["device_vs_host_cpu_max_abs"]=float(np.max(np.abs(device-host)))
    print(json.dumps(result,indent=2))
    if result["device_vs_fp32_nrms"]>=.05:raise SystemExit("device precision screen failed")

if __name__=="__main__":main()
