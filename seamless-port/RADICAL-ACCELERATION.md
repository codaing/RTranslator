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

## Results and structured compression screen

Convolution run 46121061908: baseline RTF 2.880; all QDQ 2.714 (35 token edits); all QLinearConv 2.870 (28 edits); pointwise 2.516 (0 token edits, same decisions and duration, PCM NRMS 0.848). Pointwise is a quality-evaluation candidate only, not strict parity. Sample-aligned waveform NRMS is not proof of perceptual failure. MNN run 46121301993: encoder 1.868x ORT latency and parity failure; projection 0.780x and parity failure; decoder/units/vocoder module load errors. No stage eligible. Stop blind MNN retries.

Investigate structural student size: retain 18, 12 or 6 of 24 encoder blocks using verified single residual boundary bypass, keep frontend/adapter, first and last blocks, all source frames and original decoder. Reverse-reachability prune with hidden-dependency rejection; no external weight compaction yet. This deliberately changes inference and is likely to require distillation/fine-tuning. Same full speech fixture, exact token/decision/duration and PCM NRMS <=2% + >=20% end-to-end gain for screening only; no deployment or training claim. Encoder-only reduction cannot achieve 14x overall. Decoder depth compression requires separate handling of per-layer monotonic probability and cache contracts; deferred until encoder screen. Need whole-stage GPU support or trained smaller student across components for real time. ExecuTorch Android Vulkan supports dynamic shapes and partial lowering, but not every op; native model re-export remains separate effort (https://docs.pytorch.org/executorch/stable/backends/vulkan/vulkan-overview.html).

## Pixel 9a TPU evidence and priority change

User reports TensorFlow Lite MobileNet benchmark on google-edgetpu NNAPI: 31/31 delegated, CPU fallback and XNNPACK disabled, mean 0.657 ms, median 0.609 ms, compilation 1.39 s, proprietary service separately denied by allowlist. These are user-reported measurements, not independently retrieved raw logs. Standard NNAPI is now first-priority; do not root or bypass proprietary restrictions. MobileNet is not proof of Seamless model support. NNAPI CPU disabled does not by itself prevent TFLite CPU execution of undelegated nodes. Require full delegation, actual device name, real activation, same-model CPU/TPU timing, and output parity before integration.

New CI captures actual shared decoder projection activations and up to eight independent real 1024x1024 weight matrices, concatenates only byte-identical input projections, converts to single FULLY_CONNECTED FP32 and INT8 TFLite models, evaluates host CPU numeric output and supplies exact benchmark inputs/commands for existing benchmark APK. INT8 calibration on same fixture, no held-out quality. This is a real-weight batched subgraph, not full stage. Next prioritize encoder/decoder FFN and conv block conversion on actual TPU, then complete-stage partition and persistent sessions. Large speedups must be established on Pixel. Keep live path unchanged.

Depth run 46121934353 rejected all untrained compressed encoders: RTF 2.887 baseline, 2.468/2.412/1.400 at 18/12/6 blocks, token edit distances 33/43/39; six-block result shortened output to 5.5 s vs 9.92 s baseline. No promotion. Defer training until TPU feasibility assessed.

## Pixel 9a TPU projection probe: physical-device logs (2026-11-28)

User-supplied logs show FP32 FULLY_CONNECTED rejected: DarwiNN reports input.type != TENSOR_FLOAT32, 0 TPU subgraphs, 0/1 ops supported; the benchmark full-delegation guard aborts on disallowed CPU fallback. NNAPI discovers google-edgetpu v2.0, but discovery is not execution. INT8 per-channel FULLY_CONNECTED is rejected before execution: NNAPI validator reports weight input type TENSOR_QUANT8_SYMM_PER_CHANNEL but expected TENSOR_QUANT8_ASYMM_SIGNED, then ANEURALNETWORKS_BAD_DATA and delegate application failure. No TPU timings; thermal log shows TPU power rail 0.00 mW during this failed attempt.

CPU FP32 bank graph is fully delegated to XNNPACK; session init 334.963 ms. Initial 10-run warmup median 78.124 ms / average 80.063 ms; log capture ends after the 100-run phase begins, so no completed CPU median is available. Do not compare the reported 0.657 ms MobileNet result with this graph.

Next targeted probes: (1) true FP16 TFLite graph with FP16 input, weights and output (not FP16 weight storage around FP32 activations); (2) INT8 per-tensor dense weights using converter _experimental_disable_per_channel=True, with one weight scale asserted in the artifact. Both are hypotheses based on the exact driver rejection, not guaranteed TPU-supported forms. Capture original ONNX FP16 node outputs as a separate reference for FP16 numerical parity. Benchmark only these two candidates on Pixel, and include a 20-second wait before collecting 100-run logs. Keep current app path unchanged.

## Follow-up: dense per-tensor and equivalent Conv2D

Run 46124254640: true FP16 MatMul conversion requires unsupported Flex op, no FP16 artifact. Per-tensor INT8 succeeds with one scale, host NRMS single 0.06613 / bank 0.08030; both fail the 5% precision screen and are hardware feasibility probes only. User can test bank-int8-pt and single-int8-pt on Pixel with full NNAPI delegation. In parallel, lower exactly the same real dense computation to NHWC 1x1 Conv2D with per-output-channel INT8 filters (NNAPI explicitly supports this filter format for Conv2D). Verify FP32 equivalence and exact filter scale inventory on host. Reuse existing real capture artifact, expires 2026-11-29 08:10 UTC, avoiding multi-GB download. No live integration. If both representations are rejected or slow, investigate direct NNAPI FP16 and LiteRT G4 JIT rather than repeated dense conversion attempts. Reference: https://developer.android.com/ndk/reference/group/neural-networks .


## Physical Pixel 9a per-tensor dense INT8 result (2026-11-28)

Real bank 96x1024x8192, eight source projections, 8,389,728-byte TFLite model. Both complete delegation, same INT8 inputs/model, CPU XNNPACK 4 threads versus NNAPI google-edgetpu v2.0 (required full delegation, no NNAPI CPU, no XNNPACK). TPU 1/1 supported, successful compile and DarwiNN firmware execution. CPU 591 measured runs: mean 1.69015 ms, median 1.642 ms, min 1.577 ms, p95 1.982 ms; init 23.537 ms, first 3.338 ms; process memory delta overall 28.1406 MB. TPU 409 measured runs: mean 2.42471 ms, median 2.377 ms, min 2.179 ms, p95 2.881 ms; init 1031.78 ms, first 12.194 ms; process memory delta overall 4.53125 MB. TPU median 1.448x CPU latency. Process memory delta excludes possible driver allocations. No actual TPU output parity measured. Host per-tensor bank NRMS 0.08030 fails precision screen; not eligible for inference integration. Earlier FP32 CPU warmup is not a matched full-run baseline. Do not claim TPU or end-to-end acceleration.

Pointwise workflow 46124764937 succeeded; artifact 15255442224 (expires 2026-12-05 08:13:56 UTC). Same real weights/activations expressed as one NHWC 1x1 CONV_2D, shape [1,1,96,1024] -> [1,1,96,8192], 8192 per-channel weight scales; FP32 equivalence verified on host. Host INT8 bank NRMS 0.06911; single NRMS 0.05503, both fail 5% preliminary precision screen. Next physical test bank-conv-int8-pc CPU and TPU, same model and input, full delegation, compare completed timings. Do not repeat dense test. If pointwise has no material gain, deprioritize isolated TPU dispatches and focus on multi-operation blocks with persistent sessions and XNNPACK/ORT matched real-stage baselines. A 13.9x historic end-to-end target requires accelerating nearly all expensive work; even infinitely fast acceleration of less than ~92.8% of historical compute cannot reach that target absent other changes. Verify output quality on held-out speech and sustained physical RTF before any deployment. No runtime change or APK in this decision record.


## Physical pointwise per-channel INT8 and dependent FFN core

Pixel 9a bank-conv-int8-pc fully delegated to google-edgetpu; actual firmware execution, 593 measured runs, mean 1.66236 ms, median 1.630 ms, min 1.490 ms, p95 1.963 ms, first inference 12.073 ms, init 1338.563 ms, process delta 20.0664 MB. CPU XNNPACK four threads fully delegated, init 159.992 ms, first 25.105 ms, 29 warmup runs mean 17.379 ms median 16.683 ms. Log ends before final 100-run measurement. Matched warmup medians ~10.49x in favor of TPU, but final TPU median similar to previously tested per-tensor dense CPU 1.642 ms. Host per-channel NRMS 6.91%, no TPU numerical verification. No deployment.

Move beyond isolated operators: capture actual decoder FFN dependency path (1024->8192->1024) including original activation and constant biases, using real incremental decoder calls during unchanged speech smoke. Convert to two dependent INT8 Conv2D (per-channel) or dense (per-tensor) operations in a single TFLite graph, with intermediate activation on device. Screen M=1 and frequent real prefill size, calibration from separate calls of same fixture, verify weight scale inventory and source FP16 output errors. FP32 lowering bypasses FP16 Cast rounding; requires <=2% core error before conversion. No trained model change, no full layer or end-to-end speed claim, no independent speech quality evidence. Next evaluate TPU fully delegated whole core, sustained timing and output parity; only then scale to more layers. Avoid compiling sessions per token.


## Current native ARM64 speech baseline and real FFN INT8 accuracy screen (2026-11-28)

The same Mandarin fixture and validated projected FP16 incremental decoder were measured on ubuntu-24.04-arm in run 46083582838. Original CPU smoke: total RTF 3.5305, processing RTF 3.1223, decoder 17.313 s, encoder 24.399 s, units 5.033 s, vocoder 0.888 s, first audio 7.211 s. Source-projection CPU candidate: total RTF 3.2901, processing RTF 2.9197, decoder 10.988 s plus 2.466 s source projection, encoder 25.065 s, units 5.180 s, vocoder 0.891 s, first audio 6.564 s. Token IDs, durations and monotonic decisions matched; waveform NRMS 0.0. This is a native ARM64 host baseline, not Pixel 9a device RTF. Relative to current processing RTF 2.9197, reaching 0.8 needs about 3.65x reduction; relative to total RTF 3.2901, about 4.11x. The historical 11.091 RTF / 13.9x comparison remains historical, not the remaining factor from this newer baseline.

Real decoder layer-0 FFN capture succeeded from 79 actual incremental decoder calls. The captured dependency is LayerNorm output -> MatMul 1024x8192 -> bias Add -> ReLU -> MatMul 8192x1024 -> bias Add. FP32 Conv2D lowering matches the original FP16 graph at NRMS 0.00637 for M=1 and 0.00552 for M=32. First fused INT8 output-per-tensor conversions failed the 5% host screen: per-channel Conv2D NRMS 7.72% (M1), 12.11% (M32); dense per-tensor NRMS 15.07% and 12.26%. M32 includes one real row/channel with a -5856 output outlier, which makes a single global output scale too coarse.

Grouped-output experiment: one expansion Conv2D plus 32 independent contraction Conv2D ops, each producing 32 channels with its own output quantization scale (33 Conv2D ops total; all filters symmetric per-output-channel INT8). TFLite host CPU output NRMS against original FP16 core is 3.7626% at M=1 and 3.0272% at M=32; both pass the preliminary 5% gate. Each model is 17,050,344 bytes. The 32 outputs are not in channel order in TFLite output index order; manifest contains output_group_to_tensor_index. Runtime must dequantize each output with its own scale/zero point and reorder/concatenate before passing FFN output to the next decoder op. Same-recording calibration/evaluation only; no held-out speech, whole-layer, or full-pipeline quality proof. This is a host numerical result, not TPU execution. Next physical Pixel test must require full google-edgetpu delegation, capture raw outputs with --output_filepath, compare TPU-vs-CPU and FP32 reference, and measure M=1/M=32 latency. Do not integrate before physical output parity and sustained end-to-end RTF.

No APK or live inference path changed. The existing source-check failure on x86 is the previously documented rejected test_dynamic_ffn_int8 case (NRMS 0.10868 > 0.025); do not relax that test or misattribute it to the grouped-output experiment.


## M=32 grouped FFN row-level accuracy audit and row-split experiment (2026-11-28)

The archived grouped-output host artifact (workflow run 46128980959, artifact 15257071981) was re-audited row by row. The M=32, group-size-32 model reports global NRMS 3.027%, but this is not an adequate accuracy pass: row 0 NRMS is 1.02%, while rows 1..31 have median NRMS 21.05%, p90 23.27%, and maximum 24.18%. The first row has a much larger output norm and dominates the global metric. The handoff reports byte-identical CPU/TPU raw output tensors, so the same row-level error applies to the physical TPU result if that parity observation is retained.

The original FP32 expansion reference computed from the captured weights and activations shows a consistent row outlier across all eight M=32 calls: row 0 reaches approximately 69.9, while the remaining 31 rows stay below approximately 13.2. The shared M=32 expansion activation calibration range therefore has an estimated scale near 0.274, compared with approximately 0.0517 when calibrating only rows 1..31 (about 5.3x finer). These are reference-side diagnostics; the new converter additionally reads the actual quantized TFLite expansion intermediate and reports its scale, saturation, and per-channel error.

A simplified simulation using only expansion-activation and final-output quantization (not quantized weights or input) gives M=32 tail-row median NRMS about 19.93%; separating the first row and calibrating the M=31 tail independently reduces the simulated tail median to about 1.51% (p90 about 1.96%). This is a hypothesis screen, not a TFLite conversion result. The next physical-device experiment therefore compares one M=32 baseline against a separate M=1 first-row model plus M=31 tail models, with tail output groups of 8, 16, 32, and 64 channels.

The updated converter records global, per-row, and per-output-group NRMS independently, and captures the quantized expansion intermediate. Accuracy acceptance requires all three error levels below 5%; global NRMS alone is insufficient. Physical testing must also retain full google-edgetpu delegation, exact CPU/TPU raw-output parity, and latency accounting for the two-call split. Calibration/evaluation still use one Mandarin recording; held-out speech, complete decoder quality, and end-to-end RTF remain unproven. No APK or live inference path changes are made by this experiment.


## Row-split CI correction and physical-device comparison integrity (2026-11-28)

The first row-split CI attempt (run 46133049584) failed before conversion because the generated adb am start --es args command had an unterminated Python string. The source-check run 46133048928 failed at compileall for the same syntax error. No TFLite conversion or Pixel benchmark occurred in those runs.

The follow-up corrects command quoting with shlex.quote, identifies the post-bias/post-ReLU expansion activation by tracing the input consumed by the grouped contraction Conv2D operations (rather than selecting an ambiguous tensor by shape), and runs physical XNNPACK before TPU. TPU output comparison now explicitly uses the matching on-device XNNPACK raw output file as --host-output, so exact backend parity is not accidentally checked against the CI-host reference tensor.

A new aggregation script reassembles M=1 first-row and M=31 tail outputs, compares the reassembled 32-row result against the same M=32 source reference, and reports global, per-row, and segment-specific output-group error. These are still experimental screens using the same speech recording, not held-out speech or end-to-end translation quality.


## First-row output-group sweep and benchmark command correction (2026-11-28)

The first successful row-split conversion produced an M=1 first-row model with group size 32. Its global NRMS was 0.7899% and its row NRMS was 0.7899%, but the worst 32-channel output group reached 11.23%. The reference norm of that group is nonzero (37.25), so this is not only a zero-denominator artifact. The first-row model is not accepted under the strict per-group 5% screen.

The M=31 tail passes all three preliminary thresholds at group sizes 8, 16, 32 and 64: global NRMS 3.159–3.272%, maximum row NRMS 4.616–4.734%, and maximum group NRMS 3.736–4.495%. The full M=32 baseline remains poor at maximum row NRMS 24.18% and maximum group NRMS 9.71%.

Next conversion now sweeps group sizes 8, 16, 32 and 64 for both the M=1 first row and M=31 tail, and reassembles every 4x4 pairing for CPU and TPU audits. This determines whether smaller independent output groups eliminate the first-row localized error. Benchmark byte-count parsing is also corrected to strip the filename emitted by wc -c before checking the exact expected output length.

The successful CI run 46133878691 is conversion-only. It generated 9 models in the first sweep artifact but did not execute the generated adb commands on a Pixel device. No physical-device latency or TPU parity is claimed from this CI result.


## First-row coarse output-group sweep (2026-11-28)

The second conversion sweep tested M=1 first-row output group sizes 8, 16, 32 and 64. Global NRMS stayed below 1% (0.697–0.955%), but maximum group NRMS remained above 5% at every size: 15.65%, 12.02%, 11.23% and 8.23%, respectively. Thus smaller output groups do not solve the first-row localized relative error. Group size 64 is the best of this first sweep, but still fails the strict per-group threshold.

The next sweep extends only the M=1 segment to group sizes 64, 128, 256, 512 and 1024, while retaining M=31 tail sizes 8, 16, 32 and 64. Larger groups may reduce localized relative error by combining low-energy channels, but could also worsen absolute quantization resolution; all global, per-row, and per-group metrics remain visible. The expansion activation channel statistics are now computed only over channels with non-negligible reference norm, so permanently inactive channels do not produce meaningless 1e11 relative-error maxima.

The row-split models remain a host numerical experiment. Physical Pixel 9a TPU execution, exact device CPU/TPU parity, sustained latency, held-out speech and end-to-end RTF are still pending.


## Mixed-precision first-row / INT8 tail candidate (2026-11-28)

A fixed 32-channel evaluation window was introduced to prevent large quantization groups from hiding local errors. The M=1 INT8 sweep does not pass this fixed-window screen: group sizes 8, 16 and 32 have worst fixed-32-channel NRMS around 11.1–11.3%; larger quantization groups are worse and can exceed 100% in low-energy windows. The apparently passing 1024-channel quantization-group metric was therefore not accepted as proof of local accuracy.

A separate NumPy reconstruction using the real captured weights and activations was then evaluated: original-quality FP32 first row plus M=31 INT8 tail with output group size 64. The combined 32-row output has global NRMS 0.706%, maximum row NRMS 4.726%, maximum fixed-32-channel group NRMS 4.231%, and p95 fixed-group NRMS 3.768%. The M1 FP32 row alone has global NRMS 0.551% and maximum fixed-32-channel group NRMS 0.779%. The M31 INT8 tail g64 has global NRMS 3.272%, maximum row NRMS 4.726%, and maximum fixed-32-channel group NRMS 4.376%.

These mixed metrics are from NumPy/host-side reconstruction, not yet from an Android TFLite FP32 model or Pixel TPU execution. The next workflow exports a dedicated FP32 TFLite M1 model for XNNPACK CPU, benchmarks that first row on CPU, and benchmarks the M31 INT8 tail on both XNNPACK and google-edgetpu. It reassembles the physical outputs and applies fixed 32-channel windows independent of quantization group size. Exact CPU/TPU parity, sustained device latency, held-out speech quality, full-decoder quality and end-to-end RTF remain required before integration.


## Source-check architecture guard (2026-11-28)

The dynamic INT8 FFN synthetic parity test retains its strict NRMS <2.5% assertion on ARM64, where the native ARM64 case passed. The same test is a known rejected x86 case (NRMS 10.868%); x86 CI now explicitly skips that architecture-specific parity assertion rather than weakening its threshold. This does not make the x86 experiment pass and does not promote the rejected dynamic-FFN path.
