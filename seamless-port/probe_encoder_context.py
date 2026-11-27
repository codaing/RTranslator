"""Measure whether an overlapping encoder window can reuse old source states.

Host-only diagnostic; no candidate is promoted. Compare genuine full-prefix
encoder outputs with a simulated streaming cache on identical input. No
speech is discarded: older encoded states remain in the candidate history.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np

def nrms(reference, candidate):
    if reference.shape != candidate.shape or not reference.size:
        raise ValueError("incompatible encoder tensors")
    a=reference.astype(np.float64)
    b=candidate.astype(np.float64)
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-12))

def infer_stride(a,b,delta):
    change=b-a
    if change<=0 or delta%change:
        raise ValueError("encoder time stride not established")
    return delta//change

def splice(previous,window,offset,guard,stride,expected):
    # Absolute offset and guard are in raw filterbank frames. Both must
    # align to the measured subsampling stride. Do not infer alignment from
    # waveform or trim unknown shape mismatches.
    if min(offset,guard)<0 or offset%stride or guard%stride:
        raise ValueError("unaligned window")
    start=offset//stride
    discard=guard//stride if offset else 0
    cut=start+discard
    if window.ndim!=3 or previous.ndim!=3 or window.shape[0]!=1 or previous.shape[0]!=1:
        raise ValueError("unexpected encoder layout")
    if cut>previous.shape[1] or discard>=window.shape[1]:
        raise ValueError("insufficient overlap")
    if offset==0:return window if window.shape[1]==expected else _bad_length()
    result=np.concatenate((previous[:,:cut,:],window[:,discard:,:]),axis=1)
    if result.shape[1]!=expected:return _bad_length()
    return result

def _bad_length():
    raise ValueError("window output length not aligned with full encoder")

def run(model,features,output,windows=(384,768,1152),guard=128,update=64):
    import onnxruntime as ort
    f=np.fromfile(features,dtype="<f4").reshape(-1,80)
    end=len(f)//update*update
    if end<max(windows)+update:raise ValueError("fixture too short for longest window")
    opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1
    session=ort.InferenceSession(str(model),opts,providers=["CPUExecutionProvider"])
    def encode(start,count):
        t=time.monotonic()
        y=session.run(["encoded"],{"fbank":f[None,start:count,:]})[0]
        elapsed=time.monotonic()-t
        if y.ndim!=3 or y.shape[0]!=1 or not np.isfinite(y).all():
            raise ValueError("invalid encoder output")
        return y,elapsed
    a,_=encode(0,192);b,_=encode(0,192+update)
    stride=infer_stride(a.shape[1],b.shape[1],update)
    if any(x%stride for x in (*windows,guard,update,192)):
        raise ValueError("window plan is not aligned to measured stride")
    if any(w<=guard+update for w in windows):raise ValueError("not enough window overlap")
    report={"scope":"encoder approximation diagnostic, not translation quality or Android speed",
            "stride":stride,"guard_frames":guard,"update_frames":update,
            "input_seconds":end/100,"baseline":[],"candidates":{}}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        output.write_text(json.dumps(report,indent=2)+"\n")
    # One baseline and each candidate run interleaved to limit thermal/time
    # confounding; full-prefix output stays unchanged as the reference.
    previous=None;states={}
    for w in windows:report["candidates"][str(w)]={"steps":[],"rejected":None}
    for count in range(192,end+1,update):
        full,elapsed=encode(0,count)
        row={"frames":count,"encoded_frames":full.shape[1],"seconds":elapsed}
        if previous is not None:
            if full.shape[1]-previous.shape[1]!=update//stride:
                raise ValueError("encoder stride changes across prefixes")
            shared=previous.shape[1]
            # Profile the old prefix excluding increasingly large recent tails.
            row["old_prefix_drift"]={str(g):nrms(previous[:,:shared-g,:],full[:,:shared-g,:])
                for g in (0,guard//stride,2*guard//stride) if shared>g}
        report["baseline"].append(row)
        for w in windows:
            info=report["candidates"][str(w)]
            if info["rejected"]:continue
            start=max(0,count-w)
            if not start:
                states[w]=full
                info["steps"].append({"frames":count,"window_start":0,"seconds":elapsed,
                                      "nrms":0.0,"latest_nrms":0.0})
                continue
            try:
                tail,seconds=encode(start,count)
                candidate=splice(states[w],tail,start,guard,stride,full.shape[1])
                recent=min(update//stride,full.shape[1])
                info["steps"].append({"frames":count,"window_start":start,"seconds":seconds,
                    "nrms":nrms(full,candidate),
                    "latest_nrms":nrms(full[:,-recent:,:],candidate[:,-recent:,:])})
                states[w]=candidate
            except ValueError as e:info["rejected"]=str(e)
        previous=full
        save()
        print(json.dumps({"baseline":row,"windows":{k:v["steps"][-1] for k,v in report["candidates"].items() if v["steps"]}}),flush=True)
    baseline_seconds=sum(x["seconds"] for x in report["baseline"])
    baseline_frames=sum(x["frames"] for x in report["baseline"])
    for key,info in report["candidates"].items():
        rows=info["steps"]
        if info["rejected"] or len(rows)!=len(report["baseline"]):continue
        info["summary"]={"max_nrms":max(x["nrms"] for x in rows),
                         "max_latest_nrms":max(x["latest_nrms"] for x in rows),
                         "encoder_seconds_ratio":sum(x["seconds"] for x in rows)/baseline_seconds,
                         "processed_frames_ratio":sum(x["frames"]-x["window_start"] for x in rows)/baseline_frames,
                         "exact_cache_demonstrated":False}
    save()

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("model");p.add_argument("features");p.add_argument("output")
    a=p.parse_args();run(a.model,a.features,a.output)
