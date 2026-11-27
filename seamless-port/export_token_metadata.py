"""Serialize the exact UnitY2 token/character lookup used by the checkpoint."""
import json


def write_metadata(front, tokenizer, directory):
    info = tokenizer.vocab_info
    prefix = tokenizer.create_encoder(lang="eng", mode="target").prefix_indices.tolist()
    table = []
    for idx in range(info.size):
        piece = str(tokenizer.model.index_to_token(idx))
        ids = [info.unk_idx] if idx == info.unk_idx else [
            front.char_tokenizer.model.token_to_index(ch) for ch in piece]
        punctuation = len(piece) == 1 and not piece.isalpha() and not piece.isnumeric() and piece != "▁"
        table.append([ids, punctuation, len(piece) > 1 and piece.startswith("▁")])
    payload = {"prefix":prefix,"eos":info.eos_idx,"pad":info.pad_idx,"unk":info.unk_idx,
               "comma":tokenizer.model.token_to_index(","),"table":table}
    (directory / "tokenizer.json").write_text(json.dumps(payload,separators=(",",":")),encoding="utf-8")
