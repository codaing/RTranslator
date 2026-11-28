# Radical acceleration decision record — 2026-11-28

Goal: Pixel 9a Tensor G4, continuous offline native SeamlessStreaming Mandarin→English speech, no cascade/Android TTS, no arbitrary audio cuts. Historical RTF 11.091 requires ~13.9x improvement to RTF <0.8. Need current full-pipeline device baseline. Do not infer throughput from microbenchmarks or native ARM64 server.

## Stop rules

Hand-written Vulkan single projection remains slower than ORT CPU (1.826 vs 1.149 ms incl fence). Dynamic INT8 FFN rejected (13 token edits, 2.4% end-to-end gain). Append-only encoder K/V invalidated across all 24 layers. Existing Conv calibration experiment may finish but cannot solve 14x deficit on its own. No more one-kernel tuning until a whole-stage accelerated backend shows material potential.

## Backend options

1. MNN 3.6.2: pinned public ARM64 Python package with ONNX converter; Android OpenCL/Vulkan support documented. Screen all five real graphs with unchanged baseline speech inputs, each stage in isolated process, all outputs checked, original output shapes, logits argmax and monotonic threshold. Warmed stage speed, first shape-change latency, session load and RSS. CPU host screen only; passing conversion is not GPU operator placement or speed. If promising, next build an Android MNN GPU whole-stage test with fallback reporting and real captured inputs. If unsupported converter blocks critical ops, examine targeted lowering or ExecuTorch rather than endless retries.
2. LiteRT Google Tensor SDK: official current documented targets Tensor G5/G6 (Pixel 10/11); G4 compatibility not established, beta access may be required. No assumption that Pixel 9a TPU can run arbitrary graphs. Check official compatibility before any compiler effort. SDK static shapes also matter.
3. ExecuTorch Vulkan: Android-focused delegate and upper-bound dynamic shapes; requires PyTorch re-export and real supported-op partition audit. Alternative if MNN stage gate fails. No performance guarantee.
4. Distill native model to smaller encoder/decoder with causal/chunk attention and incremental states. Existing weights cannot safely freeze full-prefix states; actual architecture adaptation and training, bilingual/math corpus and held-out speech evaluation needed. No switch to different S2S checkpoint or cascade without user approval.

References: https://github.com/alibaba/MNN ; https://pypi.org/project/mnn/ ; https://github.com/alibaba/MNN/blob/master/docs/inference/python.md ; https://developers.google.com/edge/tensor-sdk/slm-compilation ; https://developers.google.com/edge/litert/next/tensor-sdk ; https://docs.pytorch.org/executorch/stable/backends/vulkan/vulkan-overview.html ; https://huggingface.co/facebook/seamless-streaming

MNN stage screen is not yet integrated, no APK. A 20% warm stage gain with <=1% output NRMS and exact integer outputs, top IDs and policy decisions is a triage threshold only, not end-to-end promotion. CPU stage timing is sequential; dynamic input resizing/readback counted separately. Entire real speech parity and sustained physical RTF required before routing. No model weights or captures uploaded, only diagnostics. Existing failed source test (dynamic FFN x86 error 0.10868 > .025, ARM64 passed) remains an open issue in rejected experiment; do not relax gate to turn it green.
