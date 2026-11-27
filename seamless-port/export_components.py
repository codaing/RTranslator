#!/usr/bin/env python3
"""Experimental real-weight ONNX exports. Never mark an unverified graph ready."""
import argparse
import json
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("component", choices=("encoder", "decoder", "vocoder"))
    p.add_argument("output", type=Path)
    args = p.parse_args()
    import numpy as np
    import torch
    import onnx
    import onnxruntime as ort
    from torch import nn

    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float32
    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / (args.component + ".onnx")
    marker = args.output / (args.component + ".validated.json")
    marker.unlink(missing_ok=True)

    if args.component == "vocoder":
        from seamless_communication.models.vocoder.loader import load_vocoder_model
        model = load_vocoder_model("vocoder_v2", device=device, dtype=dtype).eval()

        class Wrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, units):
                return self.model(units, "eng", -1, dur_prediction=False)

        wrapper = Wrapper(model)
        names = ["units"]
        outputs = ["waveform"]
        axes = {"units": {1: "units"}, "waveform": {2: "samples"}}
        examples = [(torch.randint(0, 10000, (1, n), dtype=torch.long)
                     % model.code_generator.dict.num_embeddings,) for n in (50, 17, 113)]
    elif args.component == "encoder":
        from fairseq2.assets import asset_store
        from seamless_communication.models.unity import load_unity_config, load_unity_model
        name = "seamless_streaming_unity"
        config = load_unity_config(name)
        config.use_text_decoder = False
        config.use_text_encoder = False
        config.t2u_config = None
        card = asset_store.retrieve_card(name)
        card.field("model_config").set(config)
        model = load_unity_model(card, device=device, dtype=dtype).eval()

        class Wrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, fbank):
                encoded, _ = self.model.encode_speech(fbank, None)
                return encoded

        wrapper = Wrapper(model)
        names = ["fbank"]
        outputs = ["encoded"]
        axes = {"fbank": {1: "frames"}, "encoded": {1: "encoded_frames"}}
        examples = [(torch.randn(1, n, 80),) for n in (192, 256, 384, 512, 194)]
    else:
        from seamless_communication.models.monotonic_decoder import load_monotonic_decoder_model
        from seamless_communication.models.unity import load_unity_text_tokenizer
        model = load_monotonic_decoder_model("seamless_streaming_monotonic_decoder",
                                             device=device, dtype=dtype).eval()
        tokenizer = load_unity_text_tokenizer("seamless_streaming_unity")
        prefix = tokenizer.create_encoder(lang="eng", mode="target").prefix_indices
        dim = model.text_decoder.model_dim

        class Wrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, tokens, encoded):
                features, _, p_choose = self.model.decode(tokens, None, encoded, None)
                logits = self.model.project(features[:, -1:, :])
                return logits, features, p_choose

        wrapper = Wrapper(model)
        names = ["tokens", "encoded"]
        outputs = ["logits", "features", "p_choose"]
        axes = {
            "tokens": {1: "target_tokens"}, "encoded": {1: "source_frames"},
            "features": {1: "target_tokens"}, "p_choose": {1: "target_tokens", 2: "pooled_frames"},
        }
        examples = [(torch.cat([prefix, torch.full((n,), tokenizer.model.token_to_index(","))])
                     .long().unsqueeze(0), torch.randn(1, frames, dim))
                    for n, frames in ((3, 48), (1, 64), (7, 96))]

    wrapper.eval()
    with torch.inference_mode():
        # Capture unmodified reference outputs before applying export-only rewrites.
        references = [wrapper(*inputs) for inputs in examples]
        if args.component == "encoder":
            from shaw_export import patch_shaw_attention
            from sdpa_export import patch_sdpa
            patch_shaw_attention(model)
            patch_sdpa(model)
            for inputs, expected in zip(examples, references):
                torch.testing.assert_close(wrapper(*inputs), expected, rtol=0.002, atol=0.0002)
        torch.onnx.export(wrapper, examples[0], str(model_path), input_names=names,
                          output_names=outputs, dynamic_axes=axes, opset_version=17,
                          do_constant_folding=args.component != "encoder")
        onnx.checker.check_model(str(model_path))
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        results = []
        for case, inputs in enumerate(examples):
            expected = references[case]
            if isinstance(expected, torch.Tensor):
                expected = (expected,)
            start = time.monotonic()
            print(f"Checking {args.component} case {case}: {[list(t.shape) for t in inputs]}", flush=True)
            feeds = {n: t.numpy() for n, t in zip(names, inputs)}
            try:
                actual = session.run(outputs, feeds)
            except Exception:
                if args.component == "encoder":
                    from inspect_encoder import inspect_failure
                    inspect_failure(model_path, feeds, args.output)
                raise
            elapsed = time.monotonic() - start
            for e, a in zip(expected, actual):
                e = e.detach().cpu().numpy()
                if not np.isfinite(a).all():
                    raise RuntimeError("Non-finite ONNX output")
                np.testing.assert_allclose(a, e, rtol=0.02, atol=0.002)
            np.savez(args.output / f"{args.component}.case{case}.npz",
                     **{f"input_{n}": t.numpy() for n, t in zip(names, inputs)},
                     **{f"output_{n}": v.detach().cpu().numpy() for n, v in zip(outputs, expected)})
            results.append({"input_shapes": [list(t.shape) for t in inputs], "seconds": elapsed})
    marker.write_text(json.dumps({"component": args.component, "precision": "fp32",
                                  "parity": "PASS", "tests": results}, indent=2) + "\n")
    print(marker)


if __name__ == "__main__":
    main()
