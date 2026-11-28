"""Same-run encoder Conv quantization + full speech evaluation.

Calibrate on the existing fixture's prefixes; this is only a feasibility
screen, not a held-out accuracy or lecture test. Never publish candidate models.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from encoder_conv_int8 import calibrate,GRAPH,PROFILES
from compare_encoder_windows import waveform,distance

def main(directory,features,output):
    d=Path(directory).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True)
    feature_path=Path(features).resolve()
    f=np.fromfile(feature_path,dtype="<f4").reshape(-1,80)
    graph=d/GRAPH;backup=d/(GRAPH+".baseline")
    baseline=None;results=[]
    if backup.exists():raise RuntimeError("unexpected backup")
    for profile in ("baseline",*PROFILES):
        target=d/("encoder.conv-"+profile+".onnx")
        row={"profile":profile,"result":"FAIL","eligible":False}
        marker=d/"speech-smoke.json";marker.unlink(missing_ok=True)
        try:
            if profile!="baseline":
                report=calibrate(graph,target,f,profile)
                (out/(profile+"-conversion.json")).write_text(json.dumps(report,indent=2)+"\n")
                # Candidate external tensor locations stay beside both graphs.
                os.replace(graph,backup);os.replace(target,graph)
            args=[sys.executable,"-u",str(Path(__file__).with_name("speech_smoke.py")),
                  str(d),str(feature_path),str(out/(profile+".wav")),"--source-projections","--update-frames","64","--threads","4"]
            with (out/(profile+".log")).open("w") as log:
                code=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,timeout=1800).returncode
            data=json.loads(marker.read_text()) if marker.exists() else {}
            if data:(out/(profile+".json")).write_text(json.dumps(data,indent=2)+"\n")
            if code or data.get("result")!="PASS":raise RuntimeError("speech smoke failed, exit "+str(code))
            row.update(result="PASS",processing_rtf=data["processing_real_time_factor"],
                       stage_seconds=data["stage_seconds"],first_audio=data["first_audio"],
                       output_seconds=data["output_seconds"],max_rss_mib=data["max_rss_mib"])
            if profile=="baseline":baseline=data
            else:
                a,b=baseline["token_ids"],data["token_ids"]
                da,db=baseline["prefill_decisions"],data["prefill_decisions"]
                decision_match=len(da)==len(db) and all(x["token"]==y["token"] and (x["p_choose"]>=.5)==(y["p_choose"]>=.5) for x,y in zip(da,db))
                x,y=waveform(out/"baseline.wav"),waveform(out/(profile+".wav"))
                nrms=float(np.linalg.norm(x-y)/max(np.linalg.norm(x),1)) if x.shape==y.shape and x.size else None
                ratio=row["processing_rtf"]/baseline["processing_real_time_factor"]
                enc_ratio=data["stage_seconds"]["encoder_seconds"]/baseline["stage_seconds"]["encoder_seconds"]
                row.update(token_edit_distance=distance(a,b),token_match=a==b,decision_match=decision_match,
                           duration_match=data["output_seconds"]==baseline["output_seconds"],pcm_nrms=nrms,
                           processing_ratio=ratio,encoder_ratio=enc_ratio,target_rtf_met=row["processing_rtf"]<.8)
                row["eligible"]=bool(a==b and decision_match and row["duration_match"] and nrms is not None and nrms<=.02 and ratio<.95)
        except Exception as exc:
            row.update(result="FAIL",error=str(exc))
        finally:
            if backup.exists():os.replace(backup,graph)
            target.unlink(missing_ok=True)
            # ORT may write a new external data file with the target name;
            # never delete unrelated original external initializers.
            for p in d.glob(target.name+"*"):
                if p.is_file():p.unlink()
        results.append(row)
        (out/"comparison.json").write_text(json.dumps(results,indent=2)+"\n")
        print(json.dumps(row),flush=True)
        if profile=="baseline" and baseline is None:raise RuntimeError("baseline failed")
    if not any(r["eligible"] for r in results):raise RuntimeError("no candidate met strict speech parity and 5% end-to-end improvement")
if __name__=="__main__":main(*sys.argv[1:])
