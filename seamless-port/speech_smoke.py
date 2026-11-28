"""Four real ONNX models driven by Mandarin speech and Java filterbanks.

Supports the validated full-prefix INT8 decoder and experimental incremental
FP16 KV-cache decoder. Host smoke is not an Android speed or translation
quality benchmark.
"""
import argparse
import hashlib
import platform
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
        if token in (meta["eos"],meta["pad"]):break
        ids,punc,space=table[token]
        n=len(ids)
        if token!=meta["unk"]:
            if punc and i+1<len(tokens) and table[tokens[i+1]][2]:n+=1
            elif i>2 and table[tokens[i-1]][1] and space:n-=1
        lengths[i-1]=n
        chars.extend(ids)
    if sum(lengths)!=len(chars):raise RuntimeError("character mapping alignment")
    return np.array([chars],dtype=np.int64),np.array([lengths],dtype=np.int64)


class IncrementalDecoder:
    def __init__(self,session,meta,encoded,projected=None):
        self.session=session;self.meta=meta;self.encoded=encoded
        self.projected=projected or {}
        self.input_names={x.name for x in session.get_inputs()}
        self.caches=[np.empty((1,int(meta["heads"]),0,int(meta["head_dim"])),dtype=np.float32)
                     for _ in meta["past_names"]]

    def run(self,ids,position):
        feeds={"tokens":np.asarray([ids],dtype=np.int64),
               "encoded":self.encoded,
               "position":np.asarray([position],dtype=np.int64)}
        feeds.update({name:value for name,value in zip(self.meta["past_names"],self.caches)})
        feeds.update(self.projected)
        feeds={k:v for k,v in feeds.items() if k in self.input_names}
        values=self.session.run(self.meta["output_names"],feeds)
        logits,features,choose=values[:3]
        self.caches=values[3:]
        if not all(np.isfinite(x).all() for x in (logits,features,choose)):
            raise RuntimeError("non-finite incremental decoder output")
        token=int(logits[0,-1].argmax())
        probability=float(np.min(choose[...,-1,-1]))
        return token,probability,features


def run(directory,fbank,out,update_frames=64,session_profile='default',source_projections=False,threads=4,encoder_window=0,unit_duration_gate=False,prefill_tail=0):
    directory=Path(directory)
    meta=json.loads((directory/"tokenizer.json").read_text())
    features=np.fromfile(fbank,dtype="<f4").reshape(-1,80)
    if len(features)<400:raise ValueError("need >4 seconds of speech")
    if threads<1 or threads>8:raise ValueError("threads must be 1..8")
    opts=ort.SessionOptions();opts.intra_op_num_threads=threads;opts.inter_op_num_threads=1
    incremental_path=directory/"decoder.incremental.fp16.onnx"
    incremental=incremental_path.is_file()
    if source_projections and not incremental:raise ValueError("projection cache requires incremental decoder")
    decoder_name=("decoder.cached-step.onnx" if source_projections else incremental_path.name) if incremental else "decoder.int8.onnx"
    decoder_meta=json.loads((directory/"decoder.incremental.meta.json").read_text()) if incremental else None
    start=time.monotonic()
    from decoder_session import configure_native,configure_fp16
    decoder_opts=ort.SessionOptions();decoder_opts.intra_op_num_threads=threads;decoder_opts.inter_op_num_threads=1
    if incremental:
        configure_fp16(decoder_opts,session_profile)
        configure_fp16(opts,session_profile)
    else:
        if session_profile!='default':raise ValueError('profile applies only to incremental model')
        configure_native(decoder_opts)
    sessions=[]
    from quantize_int8 import rss_mib
    specs=[("encoder.int8.onnx",opts),(decoder_name,decoder_opts),("units.onnx",opts),("vocoder.onnx",opts)]
    for name,session_opts in specs:
        loaded=time.monotonic()
        sessions.append(ort.InferenceSession(str(directory/name),session_opts,providers=["CPUExecutionProvider"]))
        print(json.dumps({"loaded":name,"seconds":time.monotonic()-loaded,"rss_mib":rss_mib()}),flush=True)
    enc,dec,unit,voc=sessions
    projection=None
    if source_projections:
        projection=ort.InferenceSession(str(directory/"decoder.source-projections.onnx"),decoder_opts,providers=["CPUExecutionProvider"])
        print(json.dumps({"loaded":"decoder.source-projections.onnx","rss_mib":rss_mib()}),flush=True)
    duration_session=None
    if unit_duration_gate:
        duration_session=ort.InferenceSession(str(directory/"units.duration-only.onnx"),opts,
                                             providers=["CPUExecutionProvider"])
    load_time=time.monotonic()-start
    tokens=list(meta["prefix"]);duration_start=0;first_audio=None;waves=[];times=[]
    phase_totals={"encoder_seconds":0.0,"decoder_seconds":0.0,"units_seconds":0.0,"vocoder_seconds":0.0,"source_projection_seconds":0.0,"duration_gate_seconds":0.0}
    decisions=[]
    decoder_calls=0;decoder_input_tokens=0
    unit_calls=0;duration_gate_calls=0;unit_calls_skipped=0

    def predict(ids,encoded):
        nonlocal decoder_calls,decoder_input_tokens
        t0=time.monotonic()
        logits,feats,choose=dec.run(["logits","features","p_choose"],
                {"tokens":np.asarray([ids],np.int64),"encoded":encoded})
        phase_totals["decoder_seconds"]+=time.monotonic()-t0
        decoder_calls+=1;decoder_input_tokens+=len(ids)
        if not all(np.isfinite(x).all() for x in (logits,feats,choose)):
            raise RuntimeError("non-finite decoder output")
        return int(logits[0,-1].argmax()),float(choose[...,-1,-1].min()),feats

    def incremental_predict(state,ids,position):
        nonlocal decoder_calls,decoder_input_tokens
        t0=time.monotonic();result=state.run(ids,position)
        phase_totals["decoder_seconds"]+=time.monotonic()-t0
        decoder_calls+=1;decoder_input_tokens+=len(ids)
        return result

    if update_frames<32 or update_frames>160 or update_frames%32:
        raise ValueError("update_frames must be a multiple of 32 between 32 and 160")
    if encoder_window not in (0,384,768,1152):
        raise ValueError("unsupported encoder context experiment")
    if encoder_window and update_frames%16:
        raise ValueError("encoder window requires aligned cadence")
    if encoder_window and encoder_window<=128+update_frames:
        raise ValueError("insufficient overlap")
    if prefill_tail not in (0,1,4,8):
        raise ValueError("unsupported prefill reuse experiment")
    if prefill_tail and not incremental:
        raise ValueError("prefill reuse requires incremental decoder")
    from prefill_reuse import retained_prefix, retain_state
    saved_caches=None
    saved_features=None
    saved_tokens=None
    reused_prefill_tokens=0
    prefill_tokens=0
    from probe_encoder_context import splice
    encoder_history=None
    previous_count=0
    encoder_frames=0
    end=len(features)&~1
    for count in list(range(192,end,update_frames))+[end]:
        t0=time.monotonic();finished=count==end
        phase_start=time.monotonic()
        # Measured stride = 16 raw frames per encoder state. For irregular
        # EOF use the original full-prefix pass; no uncertain alignment.
        window_start=max(0,count-encoder_window) if encoder_window and count%16==0 else 0
        if window_start%16:raise RuntimeError("unaligned encoder window")
        tail=enc.run(["encoded"],{"fbank":features[None,window_start:count,:]})[0]
        if window_start:
            if encoder_history is None or previous_count%16:
                raise RuntimeError("missing aligned encoder history")
            expected=encoder_history.shape[1]+(count-previous_count)//16
            encoded=splice(encoder_history,tail,window_start,128,16,expected)
        else:
            encoded=tail
        encoder_history=encoded
        previous_count=count
        encoder_frames+=count-window_start
        enc_elapsed=time.monotonic()-phase_start;phase_totals["encoder_seconds"]+=enc_elapsed
        old=len(tokens);decoder_before=phase_totals["decoder_seconds"]
        units_elapsed=vocoder_elapsed=0.0
        state=None;feature_rows=None
        if incremental:
            projected=None
            if projection is not None:
                t_proj=time.monotonic()
                names=[o.name for o in projection.get_outputs()]
                projected=dict(zip(names,projection.run(names,{"encoded":encoded})))
                phase_totals["source_projection_seconds"]+=time.monotonic()-t_proj
            state=IncrementalDecoder(dec,decoder_meta,encoded,projected)
            # Experimental approximation: upper decoder layers depend on
            # older encoder outputs. Never represent this as exact reuse.
            # Rebuild at EOF and always recompute at least the last tail tokens.
            cut=retained_prefix(tokens,saved_tokens,prefill_tail,finished)
            if cut:
                state.caches,feature_rows=retain_state(saved_caches,saved_features,cut)
                reused_prefill_tokens+=cut
            else:
                feature_rows=[]
            prefill_tokens+=len(tokens)-cut
            tok,p,initial_features=incremental_predict(state,tokens[cut:],cut)
            decisions.append({"source_frames":count,"prefix_tokens":len(tokens),"token":tok,"p_choose":p,
                "encoded_sha256":hashlib.sha256(encoded.tobytes()).hexdigest(),"encoded_rms":float(np.sqrt(np.mean(encoded**2)))})
            feature_rows.extend(initial_features[0])
            for _ in range(50):
                if tok==meta["eos"] or (not finished and p<0.5):break
                if len(tokens)-2>=100:raise RuntimeError("100-token context limit")
                tokens.append(tok)
                tok,p,step_features=incremental_predict(state,[tokens[-1]],len(tokens)-1)
                feature_rows.append(step_features[0][0])
        else:
            for _ in range(50):
                tok,p,_=predict(tokens,encoded)
                if tok==meta["eos"] or (not finished and p<0.5):break
                if len(tokens)-2>=100:raise RuntimeError("100-token context limit")
                tokens.append(tok)

        if incremental and prefill_tail:
            # Save the accepted prefix, NOT the synthetic synthesis comma.
            saved_caches=state.caches
            saved_features=tuple(feature_rows)
            saved_tokens=tuple(tokens)
        if len(tokens)>3 and (len(tokens)>old or finished):
            ids=tokens+[meta["comma"]]
            chars,lengths=map_chars(ids,meta)
            if incremental:
                _,_,comma_features=incremental_predict(state,[meta["comma"]],len(tokens))
                feats=np.asarray([feature_rows+[comma_features[0][0]]],dtype=np.float32)
            else:
                _,_,feats=predict(ids,encoded)
            from split_unit_durations import should_decode
            unit_feeds={"features":feats,"chars":chars,"char_lens":lengths}
            predicted_durations=None
            if duration_session is not None:
                phase_start=time.monotonic()
                predicted_durations=duration_session.run(["durations"],unit_feeds)[0]
                phase_totals["duration_gate_seconds"]+=time.monotonic()-phase_start
                duration_gate_calls+=1
            if predicted_durations is not None and not should_decode(predicted_durations,duration_start,finished):
                unit_calls_skipped+=1
            else:
                phase_start=time.monotonic()
                logits,durations=unit.run(["logits","durations"],unit_feeds)
                unit_calls+=1
                if predicted_durations is not None:
                    np.testing.assert_array_equal(durations,predicted_durations)
                units_elapsed=time.monotonic()-phase_start;phase_totals["units_seconds"]+=units_elapsed
                d=durations[0]
                if duration_start>len(d) or np.any(d<0):raise RuntimeError("invalid duration context")
                offset=int(d[:duration_start].sum())
                if int(d.sum())!=logits.shape[1]:raise RuntimeError("unit duration mismatch")
                if int(d.sum())-offset>=(1 if finished else 50):
                    ids=logits[0,offset:].argmax(-1)
                    ids=np.where((ids==1)|(ids==2),5,ids)-4
                    if np.any(ids<0) or np.any(ids>=10000):raise RuntimeError("invalid acoustic unit")
                    phase_start=time.monotonic()
                    pcm=voc.run(["waveform"],{"units":ids[None,:].astype(np.int64)})[0].reshape(-1)
                    vocoder_elapsed=time.monotonic()-phase_start;phase_totals["vocoder_seconds"]+=vocoder_elapsed
                    if not np.isfinite(pcm).all() or len(pcm)==0:raise RuntimeError("invalid waveform")
                    waves.append(pcm)
                    if first_audio is None:first_audio={"source_seconds":count/100,"before_eof":not finished,"wall_seconds":time.monotonic()-start}
                    duration_start=max(0,len(d)-1)
        step_decoder=phase_totals["decoder_seconds"]-decoder_before
        times.append({"source_seconds":count/100,"seconds":time.monotonic()-t0,
            "encoder_seconds":enc_elapsed,"encoder_window_start":window_start,"decoder_seconds":step_decoder,
            "units_seconds":units_elapsed,"vocoder_seconds":vocoder_elapsed,
            "decoder_calls":decoder_calls,"decoder_input_tokens":decoder_input_tokens,
            "tokens":len(tokens)-2,"rss_mib":rss_mib()})
        print(times[-1],flush=True)
        del state,feature_rows

    pcm=np.concatenate(waves) if waves else np.array([],np.float32)
    if len(pcm):
        with wave.open(str(out),"wb") as w:
            w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000)
            w.writeframes((np.clip(pcm,-1,1)*32767).astype("<i2").tobytes())
    elapsed=time.monotonic()-start
    success=bool(first_audio and first_audio["before_eof"] and len(pcm)>=8000 and
                 np.sqrt(np.mean(pcm**2))>1e-5 and len(tokens)>4)
    result={"result":"PASS" if success else "FAIL",
       "scope":"host waveform smoke only, not translation quality or Android speed",
       "decoder_mode":"incremental-fp16-kv-cache" if incremental else "full-prefix-int8",
       "session_profile":session_profile,"source_projections":source_projections,
       "architecture":platform.machine(),"ort_version":ort.__version__,"numpy_version":np.__version__,
       "fbank_sha256":hashlib.sha256(Path(fbank).read_bytes()).hexdigest(),"prefill_decisions":decisions,
       "load_seconds":load_time,"input_seconds":end/100,"output_seconds":len(pcm)/16000,
       "intra_op_threads":threads,"processing_real_time_factor":(elapsed-load_time)/(end/100),
       "update_frames":update_frames,"update_interval_ms":update_frames*10,
       "encoder_window":encoder_window,"encoder_input_frames":encoder_frames,
       "unit_duration_gate":unit_duration_gate,"unit_calls":unit_calls,
       "prefill_tail":prefill_tail,"prefill_tokens":prefill_tokens,"reused_prefill_tokens":reused_prefill_tokens,
       "duration_gate_calls":duration_gate_calls,"unit_calls_skipped":unit_calls_skipped,
       "stage_seconds":phase_totals,"decoder_calls":decoder_calls,"decoder_input_tokens":decoder_input_tokens,
       "real_time_factor":elapsed/(end/100),"max_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
       "first_audio":first_audio,"token_ids":tokens,"steps":times}
    (directory/"speech-smoke.json").write_text(json.dumps(result,indent=2)+"\n")
    if not success:raise RuntimeError("no valid audio before end of input")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("directory");p.add_argument("features");p.add_argument("output")
    p.add_argument("--update-frames",type=int,default=64)
    p.add_argument("--session-profile",choices=("default","no-retention","basic-no-retention"),default="default")
    p.add_argument("--source-projections",action="store_true")
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--encoder-window",type=int,default=0,choices=(0,384,768,1152))
    p.add_argument("--unit-duration-gate",action="store_true")
    p.add_argument("--prefill-tail",type=int,choices=(0,1,4,8),default=0)
    a=p.parse_args();run(a.directory,a.features,a.output,a.update_frames,a.session_profile,a.source_projections,a.threads,a.encoder_window,a.unit_duration_gate,a.prefill_tail)
