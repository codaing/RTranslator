"""Isolated same-run baseline vs pruned speech comparison, no tuning on eval."""
import json,subprocess,sys
from pathlib import Path
import wave
import numpy as np

def main(source,candidate,fbank,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    reports={}
    for label,folder in (("full",Path(source)),("english",Path(candidate))):
        p=folder/"speech-smoke.json";p.unlink(missing_ok=True)
        with (output/(label+".log")).open("w") as log:
            result=subprocess.run([sys.executable,"-u",str(Path(__file__).with_name("speech_smoke.py")),
                str(folder),fbank,str(output/(label+".wav"))],stdout=log,stderr=subprocess.STDOUT,timeout=1200)
        data=json.loads(p.read_text()) if p.exists() else {}
        if data:(output/(label+".json")).write_text(json.dumps(data,indent=2)+"\n")
        reports[label]={"exit":result.returncode,"data":data}
    a,b=(reports[k]["data"] for k in ("full","english"))
    retained=set(json.loads((Path(candidate)/"english-vocab.json").read_text())["retained_ids"])
    result={"full_pass":reports["full"]["exit"]==0 and a.get("result")=="PASS",
        "english_pass":reports["english"]["exit"]==0 and b.get("result")=="PASS",
        "reference_tokens_covered":bool(a and set(a["token_ids"]).issubset(retained)),
        "token_match":bool(a and b and a["token_ids"]==b["token_ids"]),
        "duration_match":bool(a and b and a["output_seconds"]==b["output_seconds"]),
        "decision_match":False,"waveform_nrms":None}
    if a and b:
        x,y=a["prefill_decisions"],b["prefill_decisions"]
        result["decision_match"]=len(x)==len(y) and all(
            i["token"]==j["token"] and (i["p_choose"]>=.5)==(j["p_choose"]>=.5)
            for i,j in zip(x,y))
    if result["duration_match"]:
        with wave.open(str(output/"full.wav"),"rb") as w:x=np.frombuffer(w.readframes(w.getnframes()),"<i2").astype(np.float64)
        with wave.open(str(output/"english.wav"),"rb") as w:y=np.frombuffer(w.readframes(w.getnframes()),"<i2").astype(np.float64)
        if x.shape==y.shape and x.size:result["waveform_nrms"]=float(np.linalg.norm(x-y)/max(np.linalg.norm(x),1))
    result["parity"]="PASS" if all(result[k] for k in ("full_pass","english_pass","reference_tokens_covered","token_match","duration_match","decision_match")) and result["waveform_nrms"] is not None and result["waveform_nrms"]<=.02 else "FAIL"
    for label,d in (("full",a),("english",b)):
        result[label]={k:d.get(k) for k in ("real_time_factor","max_rss_mib","stage_seconds","first_audio")}
    (output/"comparison.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)
    if result["parity"]!="PASS":raise RuntimeError("vocabulary candidate did not preserve baseline speech behavior")
    (Path(candidate)/"english-vocab.validated.json").write_text(json.dumps(result,indent=2)+"\n")

if __name__=="__main__":main(*sys.argv[1:])
