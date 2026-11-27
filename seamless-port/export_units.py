"""Export the real UnitY2 acoustic unit decoder, with token-to-char mapping outside ONNX."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import onnx
import onnxruntime as ort
from torch import nn
from fairseq2.assets import asset_store
from fairseq2.nn.padding import PaddingMask
from seamless_communication.models.unity import load_unity_config, load_unity_model, load_unity_text_tokenizer


class DynamicUpsampling(nn.Module):
    """Batch-one hard upsampling expressed without Python data-dependent loops."""
    def forward(self, seqs, durations):
        ends = durations.cumsum(dim=1)
        positions = torch.arange(ends[0, -1], device=seqs.device)
        indices = (positions[:, None] >= ends[0][None, :]).sum(dim=1)
        return seqs.index_select(1, indices), ends[:, -1]


class Units(nn.Module):
    def __init__(self, t2u):
        super().__init__()
        self.t2u = t2u

    def forward(self, features, chars, char_lens):
        model = self.t2u
        front = model.decoder_frontend
        seqs, _ = model.encode(features, None)
        # All characters are valid for batch-one unpadded inference.
        seqs = front.character_level_upsampling(seqs, None, chars, char_lens)
        seqs, padding, durations = front.variance_adaptor(
            seqs, None, duration_factor=1.0, min_duration=1)
        seqs = front.forward_unit_pos_embedding(seqs, padding)
        # Batch one: duration expansion contains exactly the valid unit frames.
        # No padded keys exist, so avoid a redundant traced padding mask.
        seqs, _ = model.decoder(seqs, None, film_cond_emb=None)
        logits = model.project(seqs).logits
        return logits, durations


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    directory = args.output
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "units.validated.json"
    marker.unlink(missing_ok=True)
    name = "seamless_streaming_unity"
    cfg = load_unity_config(name)
    cfg.use_text_decoder = False
    cfg.use_text_encoder = False
    card = asset_store.retrieve_card(name)
    card.field("model_config").set(cfg)
    unity = load_unity_model(card, device=torch.device("cpu"), dtype=torch.float32).eval()
    model = unity.t2u_model.eval()
    tokenizer = load_unity_text_tokenizer(name)
    encoder = tokenizer.create_encoder(lang="eng", mode="target")
    # Input fixtures contain natural English tokens and variable-length decoder features.
    cases = []
    torch.manual_seed(0)
    with torch.inference_mode():
        for text in ("a number", "the square root of two", "the derivative of a function is zero"):
            tokens = encoder(text).reshape(1, -1)
            # Upstream frontend mutates special tokens in-place; keep separate copies.
            chars, _, lens = model.decoder_frontend.text_to_char_seqs(tokens.clone())
            features = torch.randn(1, tokens.shape[1], model.model_dim)
            out, _, durations = model(features, None, tokens.clone(), duration_factor=1.0)
            cases.append(((features, chars, lens), (out.logits, durations)))
        # Replace only the hard-upsampling implementations, preserving trained parameters.
        front = model.decoder_frontend
        front.char_length_regulator = DynamicUpsampling()
        front.variance_adaptor.length_regulator = DynamicUpsampling()
        from sdpa_export import patch_sdpa
        patch_sdpa(model)
        wrapper = Units(model).eval()
        names = ["features", "chars", "char_lens"]
        outputs = ["logits", "durations"]
        for inputs, expected in cases:
            actual = wrapper(*inputs)
            for a, e in zip(actual, expected):
                torch.testing.assert_close(a, e, rtol=0.02, atol=0.002)
        target = directory / "units.onnx"
        torch.onnx.export(wrapper, cases[0][0], str(target), input_names=names,
            output_names=outputs, opset_version=17, dynamic_axes={
                "features": {1:"tokens"}, "chars": {1:"characters"},
                "char_lens": {1:"tokens"}, "logits": {1:"units"},
                "durations": {1:"characters"}})
        onnx.checker.check_model(str(target))
        session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
        for inputs, expected in cases:
            actual = session.run(outputs, {n:t.numpy() for n,t in zip(names,inputs)})
            for a,e in zip(actual, expected):
                np.testing.assert_allclose(a,e.numpy(),rtol=0.02,atol=0.002)
            np.testing.assert_array_equal(actual[0].argmax(-1),expected[0].numpy().argmax(-1))
    # Export tokenizer metadata from the same checkpoint, avoiding guessed IDs.
    from export_token_metadata import write_metadata
    write_metadata(model.decoder_frontend, tokenizer, directory)
    marker.write_text(json.dumps({"component":"units","precision":"fp32","parity":"PASS","cases":len(cases)})+"\n")


if __name__ == "__main__":
    main()
