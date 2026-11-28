"""Render saved speech-smoke token IDs with the exact upstream tokenizer.

This is a readable diagnostic, not ASR of the generated waveform and not
a semantic accuracy score. The model's text predictions can be correct
while acoustic synthesis is incorrect.
"""
import argparse
import json
from pathlib import Path

def decode_tokens(ids,tokenizer):
    prefix=tokenizer.create_encoder(lang="eng",mode="target").prefix_indices.tolist()
    if ids[:len(prefix)]!=prefix:
        raise ValueError("English prefix mismatch")
    size=tokenizer.vocab_info.size
    payload=ids[len(prefix):]
    if any(not isinstance(i,int) or not 0<=i<size for i in payload):
        raise ValueError("invalid token ID")
    # Use the checkpoint tokenizer, not the raw SentencePiece index order.
    # The upstream streaming detokenizer replaces the word-boundary marker.
    pieces=[str(tokenizer.model.index_to_token(i)) for i in payload]
    return "".join(pieces).replace("▁"," ").strip()

def review(directory,output,tokenizer):
    directory=Path(directory)
    rows=[]
    for path in sorted(directory.glob("window-*.json")):
        data=json.loads(path.read_text())
        if data.get("result")!="PASS" or "token_ids" not in data:
            continue
        rows.append({"report":path.name,"fbank_sha256":data["fbank_sha256"],
            "encoder_window":data["encoder_window"],"text":decode_tokens(data["token_ids"],tokenizer),
            "tokens":data["token_ids"],"input_seconds":data["input_seconds"],
            "output_seconds":data["output_seconds"],"processing_rtf":data["processing_real_time_factor"]})
    if not rows or not any(r["encoder_window"]==0 for r in rows):
        raise ValueError("no baseline speech report")
    if len({r["fbank_sha256"] for r in rows})!=1:
        raise ValueError("reports use different speech fixtures")
    result={"scope":"decoder text diagnostic only; not spoken-audio or semantic validation",
            "translations":rows}
    Path(output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    for row in rows:
        print(json.dumps(row,ensure_ascii=False),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory");p.add_argument("output")
    args=p.parse_args()
    from seamless_communication.models.unity import load_unity_text_tokenizer
    review(args.directory,args.output,load_unity_text_tokenizer("seamless_streaming_unity"))
