"""Four real ONNX models driven by actual Mandarin speech and Java filterbanks.

Host-side integration smoke test mirrors the Android full-prefix policy. It is
not an Android execution test or a semantic translation accuracy benchmark.
"""
import argparse
import json
from pathlib import Path
import resource
import time
import wave
import numpy as np
import onnxruntime as ort


def map_chars(tokens,meta):
    table=meta["table"]
    chars=[]
    lengths=[0]*len(tokens)
    for i in range(2,len(tokens)):
        token=tokens[i]
        if token in (meta["eos"],meta["pad"]):
            break
        ids,punc,space=table[token]
        n=len(ids)
        if token!=meta["unk"]:
            if punc and i+1<len(tokens) and table[tokens[i+1]][2]:
                n+=1
            elif i>2 and table[tokens[i-1]][1] and space:
                n-=1
        lengths[i-1]=n
        chars.extend(ids)
    if sum(lengths)!=len(chars):
        raise RuntimeError("character mapping alignment")
    return np.array([chars],dtype=np.int64),np.array([lengths],dtype=np.int64)


def run(directory,fbank,out):
    directory=Path(directory)
    meta=json.loads((directory/"tokenizer.json").read_text())
    features=np.fromfile(fbank,dtype="<f4").reshape(-1,80)
    if len(features)<400:
        raise ValueError("need >4 seconds of speech")
    opts=ort.SessionOptions()
    opts.intra_op_num_threads=4
    opts.inter_op_num_threads=1
    start=time.monotonic()
    enc,dec,unit,voc=[ort.InferenceSession(str(directory/n),opts,providers=["CPUExecutionProvider"])
        for n in ("encoder.int8.onnx","decoder.int8.onnx","units.onnx","vocoder.onnx")]
    load_time=time.monotonic()-start
    tokens=list(meta["prefix"])
    duration_start=0
    first_audio=None
    waves=[]
    times=[]
    start=time.monotonic()

    def predict(ids,encoded):
        logits,feats,choose=dec.run(["logits","features","p_choose"],
                {"tokens":np.array([ids],np.int64),"encoded":encoded})
        if not all(np.isfinite(x).all() for x in (logits,feats,choose)):
            raise RuntimeError("non-finite decoder output")
        return int(logits[0,-1].argmax()),float(choose[...,-1,-1].min()),feats

    end=len(features)&~1
    # 320ms transport cadence, preserving the entire input prefix.
    for count in list(range(192,end,32))+[end]:
        t0=time.monotonic()
        finished=count==end
        encoded=enc.run(["encoded"],{"fbank":features[None,:count,:]})[0]
        old=len(tokens)
        for _ in range(50):
            tok,p,_=predict(tokens,encoded)
            if tok==meta["eos"] or (not finished and p<0.5):break
            if len(tokens)-2>=100:raise RuntimeError("100-token context limit")
            tokens.append(tok)
        if len(tokens)>3 and (len(tokens)>old or finished):
            ids=tokens+[meta["comma"]]
            chars,lengths=map_chars(ids,meta)
            _,_,feats=predict(ids,encoded)
            logits,durations=unit.run(["logits","durations"],{"features":feats,"chars":chars,"char_lens":lengths})
            d=durations[0]
            if duration_start>len(d) or np.any(d<0):raise RuntimeError("invalid duration context")
            offset=int(d[:duration_start].sum())
            if int(d.sum())!=logits.shape[1]:raise RuntimeError("unit duration mismatch")
            if int(d.sum())-offset>=(1 if finished else 50):
                ids=logits[0,offset:].argmax(-1)
                ids=np.where((ids==1)|(ids==2),5,ids)-4
                if np.any(ids<0) or np.any(ids>=10000):raise RuntimeError("invalid acoustic unit")
                pcm=voc.run(["waveform"],{"units":ids[None,:].astype(np.int64)})[0].reshape(-1)
                if not np.isfinite(pcm).all() or len(pcm)==0:raise RuntimeError("invalid waveform")
                waves.append(pcm)
                if first_audio is None:first_audio={"source_seconds":count/100,"before_eof":not finished,"wall_seconds":time.monotonic()-start}
                duration_start=max(0,len(d)-1)
        times.append({"source_seconds":count/100,"seconds":time.monotonic()-t0,"tokens":len(tokens)-2})
        print(times[-1],flush=True)
    pcm=np.concatenate(waves) if waves else np.array([],np.float32)
    if len(pcm):
        with wave.open(str(out),"wb") as w:
            w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000)
            w.writeframes((np.clip(pcm,-1,1)*32767).astype("<i2").tobytes())
    elapsed=time.monotonic()-start
    success=bool(first_audio and first_audio["before_eof"] and len(pcm)>=8000 and np.sqrt(np.mean(pcm**2))>1e-5 and len(tokens)>4)
    result={"result":"PASS" if success else "FAIL","scope":"host waveform smoke only, not translation quality or Android speed",
       "load_seconds":load_time,"input_seconds":end/100,"output_seconds":len(pcm)/16000,
       "real_time_factor":elapsed/(end/100),"max_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
       "first_audio":first_audio,"token_ids":tokens,"steps":times}
    (directory/"speech-smoke.json").write_text(json.dumps(result,indent=2)+"\n")
    if not success:raise RuntimeError("no valid audio before end of input")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("directory");p.add_argument("features");p.add_argument("output")
    a=p.parse_args();run(a.directory,a.features,a.output)
