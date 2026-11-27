"""Same-run ARM64 tuning, with unchanged weights and fixed audio.

A coarser cadence changes decision times and may change content. Report that
explicitly; never promote a configuration solely because it is faster.
"""
import json,subprocess,sys,os,wave
from pathlib import Path
import numpy as np

def main(directory,fbank,output):
    directory,output=Path(directory),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    results=[];baseline=None
    profiles=[(64,4),(96,4),(128,4),(160,4),(64,2)]
    if len(os.sched_getaffinity(0))>=8:profiles.append((64,8))
    for cadence,threads in profiles:
        label=f"frames{cadence}-threads{threads}"
        report=directory/"speech-smoke.json";report.unlink(missing_ok=True)
        args=[sys.executable,"-u",str(Path(__file__).with_name("speech_smoke.py")),
              str(directory),fbank,str(output/(label+".wav")),
              "--update-frames",str(cadence),"--threads",str(threads)]
        with (output/(label+".log")).open("w") as log:
            try:code=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,timeout=1200).returncode
            except subprocess.TimeoutExpired:code=124
        d=json.loads(report.read_text()) if report.exists() else {}
        if d:(output/(label+".json")).write_text(json.dumps(d,indent=2)+"\n")
        if not results:baseline=d
        item={"profile":label,"exit":code,"cadence_ms":cadence*10,"threads":threads,
              "pass":code==0 and d.get("result")=="PASS",
              "matches_reference_tokens":bool(baseline and d and d["token_ids"]==baseline["token_ids"]),
              "matches_reference_duration":bool(baseline and d and d["output_seconds"]==baseline["output_seconds"])}
        for key in ("max_rss_mib","real_time_factor","processing_real_time_factor","stage_seconds","first_audio","output_seconds","fbank_sha256"):
            item[key]=d.get(key)
        item["encoder_calls"]=len(d.get("steps",[]))
        item["waveform_nrms"]=None
        if item["matches_reference_duration"] and item["pass"] and baseline.get("result")=="PASS":
            with wave.open(str(output/"frames64-threads4.wav"),"rb") as w:
                x=np.frombuffer(w.readframes(w.getnframes()),"<i2").astype(np.float64)
            with wave.open(str(output/(label+".wav")),"rb") as w:
                y=np.frombuffer(w.readframes(w.getnframes()),"<i2").astype(np.float64)
            if x.shape==y.shape and x.size:item["waveform_nrms"]=float(np.linalg.norm(x-y)/max(np.linalg.norm(x),1))
        item["strict_fixture_match"]=bool(item["pass"] and baseline and baseline.get("result")=="PASS"
            and item["matches_reference_tokens"] and item["matches_reference_duration"]
            and item["waveform_nrms"] is not None and item["waveform_nrms"]<=.02)
        results.append(item)
        print(json.dumps(item),flush=True)
        (output/"comparison.json").write_text(json.dumps(results,indent=2)+"\n")
    if not results[0]["pass"]:raise RuntimeError("same-run baseline failed; cannot compare")
    if not any(x["strict_fixture_match"] and x["processing_real_time_factor"]<results[0]["processing_real_time_factor"] for x in results[1:]):
        raise RuntimeError("no faster strict-match profile")

if __name__=="__main__":main(*sys.argv[1:])
