"""Speed-first screening. Exact parity is reported, not assumed.

All candidates use the same projected decoder and duration preflight,
except the reference which keeps the original synthesis path. Candidate
results are experimental even if fast. No Android or model promotion.
"""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from compare_encoder_windows import distance,waveform

PROFILES=[
    ("reference",64,0,0,False),
    ("prefill-tail4",64,0,4,True),
    ("prefill-tail1",64,0,1,True),
    ("window768-tail4",64,768,4,True),
    ("window768-tail4-cadence128",128,768,4,True),
    ("window384-tail1-cadence160",160,384,1,True),
]

def main(directory,features,output):
    d,out=Path(directory),Path(output);out.mkdir(parents=True,exist_ok=True)
    results=[];baseline=None
    for tag,cadence,window,tail,gate in PROFILES:
        marker=d/"speech-smoke.json";marker.unlink(missing_ok=True)
        args=[sys.executable,"-u",str(Path(__file__).with_name("speech_smoke.py")),
              str(d),features,str(out/(tag+".wav")),"--source-projections",
              "--update-frames",str(cadence),"--encoder-window",str(window),"--prefill-tail",str(tail)]
        if gate:args.append("--unit-duration-gate")
        with (out/(tag+".log")).open("w") as log:
            try:code=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,timeout=1200).returncode
            except subprocess.TimeoutExpired:code=124
        data=json.loads(marker.read_text()) if marker.exists() else {}
        if data:(out/(tag+".json")).write_text(json.dumps(data,indent=2)+"\n")
        row={"profile":tag,"exit_code":code,"result":data.get("result","FAIL"),
             "cadence_ms":cadence*10,"encoder_window":window,"prefill_tail":tail,"duration_gate":gate,
             "processing_rtf":data.get("processing_real_time_factor"),"first_audio":data.get("first_audio"),
             "prefill_tokens":data.get("prefill_tokens"),"reused_prefill_tokens":data.get("reused_prefill_tokens"),
             "decoder_input_tokens":data.get("decoder_input_tokens"),"stage_seconds":data.get("stage_seconds"),
             "max_rss_mib":data.get("max_rss_mib"),"output_seconds":data.get("output_seconds")}
        if baseline is None:
            if code or data.get("result")!="PASS":raise RuntimeError("baseline failed")
            baseline=data
        elif code==0 and data.get("result")=="PASS":
            a,b=baseline["token_ids"][2:],data["token_ids"][2:]
            row["token_edit_distance"]=distance(a,b)
            row["token_error_rate"]=distance(a,b)/max(1,len(a))
            x,y=waveform(out/"reference.wav"),waveform(out/(tag+".wav"))
            row["pcm_exact"]=bool(np.array_equal(x,y))
            row["processing_ratio"]=data["processing_real_time_factor"]/baseline["processing_real_time_factor"]
            row["first_audio_processing_seconds"]=data["first_audio"]["wall_seconds"]-data["load_seconds"]
            row["target_rtf_met"]=bool(row["processing_rtf"]<.8)
        results.append(row)
        (out/"comparison.json").write_text(json.dumps(results,indent=2)+"\n")
        print(json.dumps(row),flush=True)
    if not any(r["exit_code"]==0 for r in results[1:]):raise RuntimeError("all speed candidates failed")
if __name__=="__main__":main(*sys.argv[1:])
