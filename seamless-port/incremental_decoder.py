"""Export and validate a one-graph incremental KV-cache decoder.

The same graph handles a full-prefix prefill (empty self-attention caches) and
single-token continuation (populated caches). Cross-attention K/V is recomputed
per token in this first experiment; self-attention K/V is cached. The cache is
rebuilt from the accepted target prefix on each new source update, because
higher-layer decoder states are conditioned on the source context.
"""
import json
from pathlib import Path
import time
import numpy as np
import torch
import onnx
import onnxruntime as ort
from torch import nn
from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.nn.transformer.multihead_attention import AttentionState
from seamless_communication.models.monotonic_decoder.loader import load_monotonic_decoder_model
from seamless_communication.models.unity import load_unity_text_tokenizer


class CatAttentionState(AttentionState):
    def __init__(self,k,v):
        self.k,self.v=k,v
    def append(self,k,v):
        self.k=torch.cat((self.k,k),dim=2)
        self.v=torch.cat((self.v,v),dim=2)
    def get(self):
        return self.k,self.v
    def reorder(self,new_order):
        self.k=self.k.index_select(0,new_order)
        self.v=self.v.index_select(0,new_order)


class IncrementalDecoder(nn.Module):
    def __init__(self,model):
        super().__init__();self.model=model
        self.layers=list(model.text_decoder.layers)
    def forward(self,tokens,encoded,position,*past):
        bag=IncrementalStateBag(4096)
        for i,layer in enumerate(self.layers):
            bag.set_state(layer.self_attn,CatAttentionState(past[2*i],past[2*i+1]))
        front=self.model.text_decoder_frontend
        embeds=front.embed(tokens)
        if front.scale!=1.0:embeds=embeds*front.scale
        indices=torch.arange(tokens.size(1),device=tokens.device,dtype=torch.long)+position.reshape(())
        embeds=embeds.float()+front.pos_encoder.freqs.index_select(0,indices)
        if front.layer_norm is not None:embeds=front.layer_norm(embeds)
        if front.dropout is not None:embeds=front.dropout(embeds)
        features,_,p_choose=self.model.text_decoder(embeds,None,encoded,None,state_bag=bag)
        logits=self.model.project(features[:,-1:,:])
        present=[]
        for layer in self.layers:
            state=bag.get_state(layer.self_attn,CatAttentionState)
            present.extend((state.k,state.v))
        return (logits,features,p_choose,*present)


def nrms(actual,expected):
    a=np.asarray(actual,dtype=np.float64);e=np.asarray(expected,dtype=np.float64)
    return float(np.sqrt(np.mean((a-e)**2))/max(float(np.sqrt(np.mean(e**2))),1e-6))


def main(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    target=directory/"decoder.incremental.fp32.onnx"
    marker=directory/"decoder.incremental.fp32.validated.json"
    marker.unlink(missing_ok=True)
    torch.set_num_threads(4);torch.manual_seed(20261127)
    model=load_monotonic_decoder_model("seamless_streaming_monotonic_decoder",
        device=torch.device("cpu"),dtype=torch.float32).eval()
    tokenizer=load_unity_text_tokenizer("seamless_streaming_unity")
    prefix=torch.as_tensor(tokenizer.create_encoder(lang="eng",mode="target").prefix_indices,dtype=torch.long).reshape(1,-1)
    layers=list(model.text_decoder.layers)
    heads=layers[0].self_attn.num_heads
    dim=model.text_decoder.model_dim//heads
    names=["tokens","encoded","position"]
    past_names=[];present_names=[]
    for i in range(len(layers)):
        past_names.extend((f"past_k_{i:02d}",f"past_v_{i:02d}"))
        present_names.extend((f"present_k_{i:02d}",f"present_v_{i:02d}"))
    output_names=["logits","features","p_choose"]+present_names
    wrapper=IncrementalDecoder(model).eval()
    sample_tokens=torch.cat((prefix,torch.tensor([[101]],dtype=torch.long)),dim=1)
    sample_encoded=torch.randn(1,64,model.text_decoder.model_dim)
    empty=[torch.empty((1,heads,0,dim),dtype=torch.float32) for _ in past_names]
    dynamic={"tokens":{1:"current_tokens"},"encoded":{1:"source_frames"},
             "features":{1:"current_tokens"},"p_choose":{1:"current_tokens",2:"source_frames"}}
    for n in past_names:dynamic[n]={2:"past_target_tokens"}
    for n in present_names:dynamic[n]={2:"total_target_tokens"}
    start=time.monotonic()
    torch.onnx.export(wrapper,(sample_tokens,sample_encoded,torch.tensor([0],dtype=torch.long),*empty),
        str(target),input_names=names+past_names,output_names=output_names,
        dynamic_axes=dynamic,opset_version=17,do_constant_folding=False,external_data=True)
    onnx.checker.check_model(str(target))
    print(f"Exported incremental graph in {time.monotonic()-start:.1f}s ({target.stat().st_size:,} graph bytes)",flush=True)
    opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1
    session=ort.InferenceSession(str(target),sess_options=opts,providers=["CPUExecutionProvider"])
    vocab=model.final_proj.weight.shape[0]
    all_metrics=[]
    for case,frames in enumerate((48,64,96)):
        tail=torch.randint(100,vocab-100,(1,6),dtype=torch.long)
        full=torch.cat((prefix,tail),dim=1)
        encoded=torch.randn(1,frames,model.text_decoder.model_dim)
        encoded_np=encoded.numpy()
        caches=[np.empty((1,heads,0,dim),dtype=np.float32) for _ in past_names]
        fixture={"encoded":encoded_np,"steps":np.array([1+tail.shape[1]],dtype=np.int64)}
        case_metrics=[]
        for step in range(1+tail.shape[1]):
            if step==0:
                chunk=prefix;position=0;reference_tokens=prefix
            else:
                chunk=tail[:,step-1:step];position=prefix.shape[1]+step-1
                reference_tokens=full[:,:prefix.shape[1]+step]
            with torch.inference_mode():
                ref_features,_,ref_choose=model.decode(reference_tokens,None,encoded,None)
                ref_logits=model.project(ref_features[:,-1:,:])
                exp_features=ref_features if step==0 else ref_features[:,-1:,:]
                exp_choose=ref_choose if step==0 else ref_choose[:,-1:,:]
            feeds={"tokens":chunk.numpy(),"encoded":encoded_np,"position":np.array([position],dtype=np.int64)}
            feeds.update({n:v for n,v in zip(past_names,caches)})
            t0=time.monotonic();values=session.run(output_names,feeds);elapsed=time.monotonic()-t0
            actual_logits,actual_features,actual_choose=values[:3]
            metrics={"case":case,"step":step,"seconds":elapsed,
                "logits_nrms":nrms(actual_logits,ref_logits.numpy()),
                "features_nrms":nrms(actual_features,exp_features.numpy()),
                "p_choose_nrms":nrms(actual_choose,exp_choose.numpy()),
                "top_token_agreement":bool(np.array_equal(actual_logits.argmax(-1),ref_logits.numpy().argmax(-1))),
                "read_write_agreement":bool((np.min(actual_choose[...,-1,-1])>=0.5)==(np.min(exp_choose.numpy()[...,-1,-1])>=0.5))}
            if max(metrics[k] for k in ("logits_nrms","features_nrms","p_choose_nrms"))>0.005 or not metrics["top_token_agreement"] or not metrics["read_write_agreement"]:
                raise RuntimeError(f"incremental FP32 parity failed: {metrics}")
            case_metrics.append(metrics)
            fixture[f"input_tokens_{step}"]=chunk.numpy()
            fixture[f"input_position_{step}"]=np.array([position],dtype=np.int64)
            fixture[f"expected_logits_{step}"]=ref_logits.numpy()
            fixture[f"expected_features_{step}"]=exp_features.numpy()
            fixture[f"expected_p_choose_{step}"]=exp_choose.numpy()
            caches=values[3:]
            print(json.dumps(metrics),flush=True)
        np.savez_compressed(directory/f"decoder.incremental.case{case}.npz",**fixture)
        all_metrics.extend(case_metrics)
    meta={"layers":len(layers),"heads":heads,"head_dim":dim,"model_dim":model.text_decoder.model_dim,
          "past_names":past_names,"present_names":present_names,"output_names":output_names,
          "inputs":names+past_names,"precision":"fp32"}
    (directory/"decoder.incremental.meta.json").write_text(json.dumps(meta,indent=2)+"\n")
    marker.write_text(json.dumps({"component":"decoder","precision":"fp32-incremental-kv","parity":"PASS",
        "graph_bytes":target.stat().st_size,"layers":len(layers),"tests":all_metrics},indent=2)+"\n")
    print(marker)

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("directory",type=Path)
    a=p.parse_args();main(a.directory)
