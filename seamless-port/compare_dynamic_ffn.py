"""Same-run full speech comparison of selective dynamic INT8 decoder FFN.

Candidates are never installed or published as production model packs. Baseline
keeps full prefix encoder, fresh source-conditioned decoder caches, all
vocabulary IDs, 64-frame cadence and the unmodified acoustic chain.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from dynamic_ffn_int8 import create,GRAPH
from compare_encoder_windows import waveform,distance

def main(directory,features,output):
    d=Path(directory);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    graph=d/GRAPH
    baseline=None;results=[]
    def write():
        (out/"comparison.json").write_text(json.dumps(results,indent=2)+"\n")
    for profile in ("baseline","all","expand","contract"):
        candidate=weights=None;backup=d/(GRAPH+".baseline")
        row={"profile":profile,"result":"FAIL","eligible":False}
        marker=d/"speech-smoke.json";marker.unlink(missing_ok=True)
        try:
            if profile!="baseline":
                candidate,weights,conversion=create(d,profile)
                (out/(profile+"-conversion.json")).write_text(json.dumps(conversion,indent=2)+"\n")
                if backup.exists():raise RuntimeError("unexpected existing decoder backup")
                os.replace(graph,backup)
                os.replace(candidate,graph)
            args=[sys.executable,"-u",str(Path(__file__).with_name("speech_smoke.py")),str(d),str(features),
                  str(out/(profile+".wav")),"--source-projections","--update-frames","64","--threads","4"]
            with (out/(profile+".log")).open("w") as f:
                code=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,timeout=1800).returncode
            data=json.loads(marker.read_text()) if marker.exists() else {}
            if data:(out/(profile+".json")).write_text(json.dumps(data,indent=2)+"\n")
            if code or data.get("result")!="PASS":raise RuntimeError("speech smoke failed, exit "+str(code))
            row.update(result="PASS",processing_rtf=data["processing_real_time_factor"],stage_seconds=data["stage_seconds"],
                       first_audio=data["first_audio"],output_seconds=data["output_seconds"],max_rss_mib=data["max_rss_mib"])
            if profile=="baseline":baseline=data
            else:
                a,b=baseline["token_ids"],data["token_ids"]
                x,y=waveform(out/"baseline.wav"),waveform(out/(profile+".wav"))
                da,db=baseline["prefill_decisions"],data["prefill_decisions"]
                decisions=len(da)==len(db) and all(a["token"]==b["token"] and (a["p_choose"]>=.5)==(b["p_choose"]>=.5) for a,b in zip(da,db))
                nrms=float(np.linalg.norm(x.astype(np.float64)-y.astype(np.float64))/max(np.linalg.norm(x.astype(np.float64)),1)) if x.shape==y.shape and x.size else None
                ratio=row["processing_rtf"]/baseline["processing_real_time_factor"]
                row.update(token_edit_distance=distance(a,b),token_match=a==b,decision_match=decisions,
                           duration_match=data["output_seconds"]==baseline["output_seconds"],pcm_nrms=nrms,processing_ratio=ratio,
                           target_rtf_met=row["processing_rtf"]<.8)
                row["eligible"]=bool(a==b and decisions and row["duration_match"] and nrms is not None and nrms<=.02 and ratio<.95)
        except Exception as exc:
            row.update(result="FAIL",error=str(exc))
        finally:
            if backup.exists():os.replace(backup,graph)
            if candidate:candidate.unlink(missing_ok=True)
            if weights:weights.unlink(missing_ok=True)
        results.append(row);write();print(json.dumps(row),flush=True)
        if profile=="baseline" and baseline is None:raise RuntimeError("baseline failed")
    if not any(r["eligible"] for r in results):raise RuntimeError("no candidate passed strict parity and 5% end-to-end speed gate")

if __name__=="__main__":main(*sys.argv[1:])
