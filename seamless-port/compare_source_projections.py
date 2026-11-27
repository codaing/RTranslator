"""Compare baseline and hoisted decoder on the same runner, retaining failures."""
import json
from pathlib import Path
import subprocess
import sys
import wave
import numpy as np

def main(directory,features,output):
    directory,output=Path(directory),Path(output)
    baseline=None;results=[]
    for mode in ("original","projected"):
        report=directory/"speech-smoke.json";report.unlink(missing_ok=True)
        args=[sys.executable,"-u",str(Path(__file__).with_name("speech_smoke.py")),
              str(directory),features,str(output/(mode+".wav"))]
        if mode=="projected":args.append("--source-projections")
        with (output/(mode+".log")).open("w") as f:
            code=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,timeout=1200).returncode
        data=json.loads(report.read_text()) if report.exists() else {}
        if data:(output/(mode+".json")).write_text(json.dumps(data,indent=2)+"\n")
        if mode=="original":baseline=data
        info={"mode":mode,"exit_code":code,"result":data.get("result","FAIL"),
              "max_rss_mib":data.get("max_rss_mib"),"rtf":data.get("real_time_factor"),
              "stage_seconds":data.get("stage_seconds"),"first_audio":data.get("first_audio"),
              "architecture":data.get("architecture"),"fbank_sha256":data.get("fbank_sha256")}
        if mode=="projected" and baseline and data:
            info["token_match"]=data["token_ids"]==baseline["token_ids"]
            info["duration_match"]=data["output_seconds"]==baseline["output_seconds"]
            a,b=baseline["prefill_decisions"],data["prefill_decisions"]
            info["decision_match"]=len(a)==len(b) and all(
                x["token"]==y["token"] and (x["p_choose"]>=0.5)==(y["p_choose"]>=0.5)
                for x,y in zip(a,b))
            info["eligible"]=bool(code==0 and baseline["result"]=="PASS" and info["token_match"] and info["duration_match"] and info["decision_match"])
        results.append(info)
        print(json.dumps(info),flush=True)
        (output/"comparison.json").write_text(json.dumps(results,indent=2)+"\n")
    if not results[-1].get("eligible"):raise RuntimeError("projection experiment not eligible for promotion")

if __name__=="__main__":main(*sys.argv[1:])
