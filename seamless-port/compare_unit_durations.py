"""Compare duration preflight with full UnitY2 on a fixed real speech fixture."""
import json
from pathlib import Path
import subprocess
import sys
from compare_encoder_windows import waveform
import numpy as np

def main(directory,features,output):
    d,out=Path(directory),Path(output);out.mkdir(parents=True,exist_ok=True)
    results=[];baseline=None
    for mode in ("original","gated"):
        marker=d/"speech-smoke.json";marker.unlink(missing_ok=True)
        args=[sys.executable,"-u",str(Path(__file__).with_name("speech_smoke.py")),
              str(d),features,str(out/(mode+".wav")),"--source-projections"]
        if mode=="gated":args.append("--unit-duration-gate")
        with (out/(mode+".log")).open("w") as f:
            code=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,timeout=1200).returncode
        data=json.loads(marker.read_text()) if marker.exists() else {}
        if data:(out/(mode+".json")).write_text(json.dumps(data,indent=2)+"\n")
        info={"mode":mode,"exit_code":code,"result":data.get("result","FAIL"),
              "processing_rtf":data.get("processing_real_time_factor"),"stage_seconds":data.get("stage_seconds"),
              "unit_calls":data.get("unit_calls"),"gate_calls":data.get("duration_gate_calls"),
              "skipped":data.get("unit_calls_skipped"),"max_rss_mib":data.get("max_rss_mib")}
        if mode=="original":baseline=data
        elif code==0 and baseline.get("result")=="PASS" and data.get("result")=="PASS":
            x,y=waveform(out/"original.wav"),waveform(out/"gated.wav")
            info["pcm_exact"]=bool(np.array_equal(x,y))
            info["token_match"]=baseline["token_ids"]==data["token_ids"]
            info["decision_match"]=baseline["prefill_decisions"]==data["prefill_decisions"]
            info["unit_time_ratio"]=(data["stage_seconds"]["units_seconds"]+data["stage_seconds"]["duration_gate_seconds"])/baseline["stage_seconds"]["units_seconds"]
            info["eligible"]=bool(info["pcm_exact"] and info["token_match"] and info["decision_match"]
                and info["skipped"]>0 and info["unit_time_ratio"]<1
                and info["processing_rtf"]<baseline["processing_real_time_factor"])
        results.append(info)
        (out/"comparison.json").write_text(json.dumps(results,indent=2)+"\n")
        print(json.dumps(info),flush=True)
        if mode=="original" and (code or data.get("result")!="PASS"):raise RuntimeError("baseline failed")
    if not results[-1].get("eligible"):raise RuntimeError("duration preflight not eligible")
if __name__=="__main__":main(*sys.argv[1:])
