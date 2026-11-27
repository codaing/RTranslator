"""End-to-end window approximation diagnostic. No deployment/promotion."""
import json
from pathlib import Path
import subprocess
import sys
import wave
import numpy as np

def distance(a,b):
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        nxt=[i]
        for j,y in enumerate(b,1):
            nxt.append(min(row[j]+1,nxt[-1]+1,row[j-1]+(x!=y)))
        row=nxt
    return row[-1]

def waveform(path):
    with wave.open(str(path),"rb") as f:
        return np.frombuffer(f.readframes(f.getnframes()),"<i2").astype(np.float64)

def main(directory,features,output):
    directory,output=Path(directory),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    results=[];baseline=None
    for window in (0,1152,768,384):
        tag=f"window-{window}"
        marker=directory/"speech-smoke.json";marker.unlink(missing_ok=True)
        args=[sys.executable,"-u",str(Path(__file__).with_name("speech_smoke.py")),
              str(directory),features,str(output/(tag+".wav")),
              "--source-projections","--encoder-window",str(window)]
        with (output/(tag+".log")).open("w") as log:
            try:code=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,timeout=1200).returncode
            except subprocess.TimeoutExpired:code=124
        data=json.loads(marker.read_text()) if marker.exists() else {}
        if data:(output/(tag+".json")).write_text(json.dumps(data,indent=2)+"\n")
        info={"window":window,"exit_code":code,"result":data.get("result","FAIL"),
              "processing_rtf":data.get("processing_real_time_factor"),
              "stage_seconds":data.get("stage_seconds"),"encoder_input_frames":data.get("encoder_input_frames"),
              "output_seconds":data.get("output_seconds"),"first_audio":data.get("first_audio"),
              "max_rss_mib":data.get("max_rss_mib"),"promotion":False}
        if window==0:
            if code or data.get("result")!="PASS":
                raise RuntimeError("baseline speech test failed; see window-0.log")
            baseline=data
        elif code==0 and data.get("result")=="PASS":
            a,b=baseline["token_ids"][2:],data["token_ids"][2:]
            info["token_match"]=a==b
            info["token_edit_distance"]=distance(a,b)
            info["token_error_rate"]=distance(a,b)/max(1,len(a))
            da,db=baseline["prefill_decisions"],data["prefill_decisions"]
            info["decision_match"]=len(da)==len(db) and all(
                x["source_frames"]==y["source_frames"] and x["token"]==y["token"]
                and (x["p_choose"]>=.5)==(y["p_choose"]>=.5) for x,y in zip(da,db))
            info["duration_match"]=data["output_seconds"]==baseline["output_seconds"]
            x,y=waveform(output/"window-0.wav"),waveform(output/(tag+".wav"))
            info["waveform_nrms"]=float(np.linalg.norm(x-y)/max(np.linalg.norm(x),1)) if x.shape==y.shape else None
            info["processing_ratio"]=data["processing_real_time_factor"]/baseline["processing_real_time_factor"]
            info["encoder_ratio"]=data["stage_seconds"]["encoder_seconds"]/baseline["stage_seconds"]["encoder_seconds"]
        results.append(info)
        (output/"comparison.json").write_text(json.dumps(results,indent=2)+"\n")
        print(json.dumps(info),flush=True)
    # A completed diagnostic does not mean the approximation passed quality.
    if not any(x["exit_code"]==0 for x in results[1:]):
        raise RuntimeError("no candidate completed; review logs")

if __name__=="__main__":main(*sys.argv[1:])
