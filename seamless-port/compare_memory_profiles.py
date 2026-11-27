"""Run isolated process comparisons on the same host and recording.

Report failures, token agreement, waveform length and memory. A smaller
peak is not sufficient to promote a profile if speech behavior changes.
"""
import json
from pathlib import Path
import subprocess
import sys

def main(directory,features,output):
    directory,output=Path(directory),Path(output)
    results=[]
    baseline=None
    for profile in ("default","no-retention","basic-no-retention"):
        report=directory/"speech-smoke.json"
        report.unlink(missing_ok=True)
        cmd=[sys.executable,"-u",str(Path(__file__).with_name("speech_smoke.py")),
             str(directory),features,str(output/(profile+".wav")),"--update-frames","64",
             "--session-profile",profile]
        with (output/(profile+".log")).open("w") as log:
            try:
                run=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
                code=run.returncode
            except subprocess.TimeoutExpired:
                code=124
        data=json.loads(report.read_text()) if report.exists() else {}
        if data:(output/(profile+".json")).write_text(json.dumps(data,indent=2)+"\n")
        entry={"profile":profile,"exit_code":code,"result":data.get("result","FAIL"),
               "max_rss_mib":data.get("max_rss_mib"),"real_time_factor":data.get("real_time_factor"),
               "stage_seconds":data.get("stage_seconds"),"first_audio":data.get("first_audio"),
               "output_seconds":data.get("output_seconds")}
        if profile=="default" and code==0 and entry["result"]=="PASS":baseline=data
        entry["matches_default_tokens"]=bool(baseline and data.get("token_ids")==baseline["token_ids"])
        entry["matches_default_duration"]=bool(baseline and data.get("output_seconds")==baseline["output_seconds"])
        entry["eligible"]=bool(code==0 and entry["result"]=="PASS" and entry["matches_default_tokens"] and entry["matches_default_duration"])
        results.append(entry)
        print(json.dumps(entry),flush=True)
        (output/"comparison.json").write_text(json.dumps(results,indent=2)+"\n")
    if not baseline:raise RuntimeError("same-run default reference failed; no promotion")
    if not any(x["eligible"] and x["profile"]!="default" for x in results):
        raise RuntimeError("no alternative profile passed")

if __name__=="__main__":main(*sys.argv[1:])
