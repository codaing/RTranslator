# SeamlessStreaming Android port — phase 1 (not an Android translator yet)

Goal: continuous Mandarin lecture audio to English speech on Pixel 9a, with no Android TextToSpeech, no Whisper/HY-MT cascade, no cloud service, and no fixed utterance cutoff.

This branch now has two deliberately separate tracks: the existing RTranslator app in :app is the immediate production path (with a low-latency speech profile), while the :seamless module remains an experimental single-model Mandarin-to-English research app. The experimental Seamless module is not yet a production translator. It contains an offline reference runner for the actual Meta SeamlessStreaming S2ST pipeline, an ONNX component-export experiment with numerical checks, and an Android English unit-vocoder adapter. It does not yet wire the model into the Android UI. The experimental release APK built on this branch compiles the existing Android code and vocoder adapter only; it is not a SeamlessStreaming translator.

## Upstream and model constraints

Meta's streaming implementation uses a speech feature extractor, streaming-trained speech encoder (the reference re-encodes the accumulated prefix), EMMA monotonic text decoder, non-autoregressive UnitY2 acoustic-unit decoder, and HiFi-GAN vocoder. It does not require the Android TTS engine. The original model is multilingual and predicts English; this is not an input-language classifier. Feed Mandarin audio; strict rejection of non-Mandarin speech would require a separate gate.

Upstream: https://github.com/facebookresearch/seamless_communication
Model: https://huggingface.co/facebook/seamless-streaming (CC BY-NC 4.0; personal/noncommercial use).
The official runtime is Python/PyTorch/fairseq2; Android is not a supported deployment target. Meta warns against CPU real-time inference. Limiting output to English does not remove the multilingual encoder/decoder weights.

## Phase 1: offline reference on a development computer

Use Linux x86-64 with an NVIDIA GPU, or CPU for a slow functional check. Follow Meta's install instructions for a compatible PyTorch/fairseq2 pair; use Python 3.10 in a clean environment. The following is an example for CUDA 11.8 (use your hardware-compatible variant):
```sh
git clone https://github.com/facebookresearch/seamless_communication.git
cd seamless_communication
git checkout 9a081e935c29c6b1e0e05bcbf3fcb623ad306524
python -m pip install 'torch==2.1.1' 'torchaudio==2.1.1' --index-url https://download.pytorch.org/whl/cu118
python -m pip install 'fairseq2==0.2.*' --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.1.1/cu118
python -m pip install -e .
python -m pip install 'numpy<2' 'onnx==1.16.2' 'onnxruntime==1.18.1'
```
The upstream fairseq2 wheel availability must be checked for your OS and Python version. If the historical index is unavailable, use the compatible installation described in Meta's README. The first model initialization downloads the weights; inference after caching is local.

Convert your own lecture sample to 16 kHz mono PCM WAV:
```sh
ffmpeg -i lecture.mp4 -ac 1 -ar 16000 -c:a pcm_s16le lecture.wav
python seamless-port/reference_stream.py lecture.wav translated.wav --device cuda:0
```
Run from the RTranslator root (with the Meta package installed in the same environment). The script sends consecutive 320 ms PCM frames into one persistent SimulEval agent. It never marks a frame finished until EOF, writes returned speech immediately, and reports first-output latency and real-time factor. This is a file replay, not live Android capture; --realtime additionally paces input like a microphone. Long-session context growth and early resets in Meta's reference policy must be measured before a lecture-length port.

## Phase 2: ONNX feasibility

```sh
python seamless-port/export_components.py vocoder ./seamless-models
python seamless-port/export_components.py encoder ./seamless-models
python seamless-port/export_components.py decoder ./seamless-models
```
These experiments use actual upstream weights, batch one and variable time dimensions. The exporter checks each candidate against PyTorch on multiple input lengths, using the CPU ONNX Runtime provider, and publishes a PASS marker only after successful numerical comparison. An export or parity failure is a blocker, not evidence that the component works. The encoder/decoder experiments need substantial workstation RAM; they are not optimized mobile graphs. The decoder uses a complete target prefix for each call, not KV-cache export. The Python agent remains the reference for read/write decisions and synthesis alignment.

The Android EnglishUnitVocoder class accepts only a successfully exported English vocoder graph and acoustic unit IDs from the future UnitY2 runtime, returning 16 kHz PCM. It uses ONNX Runtime already included in the app, not Android TTS. It is not hooked up to the existing translator.

## INT8 deployment target (experimental)

Quantization targets signed INT8 weights for constant-weight MatMul/Gemm operations in the encoder and monotonic decoder. Dynamic activation quantization may add runtime overhead; other operators remain floating point. Keep the vocoder FP32 initially, and measure rather than assume speedup. The actual model size will differ from the theoretical parameter estimate.

After each FP32 export succeeds:
```sh
python seamless-port/quantize_int8.py encoder ./seamless-models
python seamless-port/quantize_int8.py decoder ./seamless-models
```
The quantizer requires the FP32 validation report and saved test inputs/reference outputs. It uses ONNX Runtime dynamic QInt8 per-channel quantization, writes external tensor data, and checks every output against the FP32 references. For decoder logits it checks top-token agreement, and for monotonic attention it checks the read/write decision at threshold 0.5. Synthetic component fixtures are a first gate only, not a Mandarin speech quality benchmark. Rejected graphs must not be deployed. Run real-audio parity, sustained latency and memory tests before selecting a mobile execution provider.

The source-check workflow also exercises the quantizer on a tiny synthetic ONNX model, with no checkpoint download. The separate experimental APK workflow compiles the Android vocoder adapter but does not package the model weights or enable SeamlessStreaming inference.

Remaining work, in order:
1. Validate encoder, decoder and vocoder exports on real checkpoints. Verify dynamic shapes, numerical parity and output audio.
2. Port UnitY2 text-token to character mapping, hard upsampling/duration prediction and acoustic-unit decoder. Its Python loops and data-dependent lengths are not safe to simply trace. Export the tensor-only stages; validate acoustic units and alignment against the reference.
3. Port the filterbank extractor (25 ms windows/10 ms hop, 80 mel bins), EMMA read/write scheduling and streaming state to Android. Preserve context; define a semantic/context-limit reset strategy for long lectures rather than fixed five-second cuts.
4. Quantize only after FP32 parity; test memory and real-time factor on Pixel 9a, including sustained thermal load. Require output before source EOF, no missing/duplicated words, and real-time factor below one on sustained Mandarin input.
5. Replace the UI/model downloader with a single Mandarin-to-English mic/headset screen only once the above runs locally; then remove legacy ASR/translation/TTS on this branch.

No model weights are committed. No backend server, network inference, or fallback TTS is included in this work. No successful model export or Android performance claim is made yet.

## Dedicated Android test module

`:seamless` now builds a separate application (nie.translator.seamless) with only ONNX Runtime, no legacy model downloader/TTS/Whisper code, compressed JNI libraries and ARM64 packaging. It imports a standalone vocoder and plays synthetic acoustic units. It is explicitly not a translator yet. The APK workflow checks ELF/ZIP alignment, installs on Android 16 and runs a real Android ONNX interface test using a tiny synthetic graph. The emulator test does not prove model translation or 16 KB physical-device runtime compatibility.

## Experimental integrated streaming path

Import a flat ZIP containing encoder.int8.onnx, decoder.int8.onnx, units.onnx, vocoder.onnx, their external data files, and tokenizer.json produced by export_token_metadata.py. No network permissions or Android TTS. This is an integration prototype, NOT a validated translator. No distributable model pack is provided yet. Continuous capture uses 320 ms transport frames (not independent utterances), accumulating filterbank context. The first encoder pass starts at 192 feature frames. Full-prefix re-encoding and full-prefix decoder runs are costly; the prototype explicitly stops at a 60-second source budget rather than silently dropping audio. It needs calibrated real-waveform filterbank parity, full pipeline tests with real speech, a Mandarin-only source gate, safe long-context rollover and physical-device performance validation. A source-language selector is not implemented: the multilingual encoder may translate non-Mandarin input too. Playback uses an attached headphone route and stops if it changes. Foreground-only operation. INT8 applies only to encoder/decoder; units/vocoder remain FP32. No ready-for-use claims.

## Host integration and model packaging

The real-model workflow has a manual `package_models` input. It is disabled by default to avoid repeatedly transferring gigabytes of weights on public CI. Enable it for personal noncommercial testing only (Meta CC BY-NC 4.0). All component parity gates must pass. The integrated job downloads pinned Mandarin sample WAVs from csukuangfj/sherpa-onnx-paraformer-zh-2023-09-14, concatenates them, compares the actual Java filterbanks to fairseq2, then exercises all four real ONNX models in a host streaming smoke test. It requires non-silent generated audio before EOF, records first-output delay, per-step timings, peak RSS and real-time factor, and uploads the output WAV for listening. These are NOT semantic accuracy, Android execution, or Pixel speed tests. The pack builder checks component reports, resolves external tensors, rejects conflicting names, includes model attribution/license and SHA-256 hashes, and only emits the import ZIP after the speech smoke gate. GitHub artifacts may exceed runner disk or storage quotas; no package has been verified yet. No fixed-duration reset added: full-prefix rollover cannot safely discard audio without proving source consumption. The explicit 60-second/100-token stop remains until rollover parity tests can establish safe behavior.

A push with `[package-models]` in the commit message also requests the gated packaging job, without depending on workflow_dispatch availability on the default branch. Routine pushes still upload reports only.

## Experimental context rollover and device diagnostics

Opt-in only: when context is at least 8 seconds, the current decoder predicts EOS with minimum-head probability >=0.5, at least two target tokens exist, and the most recent PCM contains >=600 ms of quiet audio (RMS <=250), attempt final unit synthesis and then reset at the encoded even-frame boundary, retaining an odd frame and filterbank residual. EOS is not proof of source coverage, so this can omit/repeat material or lose cross-sentence context. Disabled by default. No timer-driven cuts; uninterrupted speech with no eligible boundary still hits the 60-second/100-token safety limit. Rollover needs real speech coverage tests before use in lectures. Live display reports input/capture backlog, generated-not-submitted queue, processing RTF, first generated audio (not semantic translation delay or actual headphone latency), PSS, current context and reset count. Model import requires the pack's SHA256 manifest and checks every extracted file before activation; this detects corruption, not malicious replacement of both files and manifest. Existing packs lacking the manifest are rejected.

## Selective decoder INT8

Full decoder dynamic INT8 previously failed real-weight fixture parity (logits normalized RMS 0.349 versus maximum 0.10). The next experiment uses explicit FFN inner/output projections only; attention (including monotonic decisions), token embeddings and final vocabulary projection stay FP32. The encoder keeps its previously validated policy. All original numerical gates remain unchanged. Reports include node-level selection, initializer storage by dtype, all fixture metrics even on accuracy failure, and 30-second phase/RSS heartbeats. This mixed-precision decoder may be substantially larger than full INT8 and is not yet validated for Pixel RAM. The `.int8.onnx` filename denotes partial dynamic INT8, not all-int8 weights.

## Weight-only decoder experiment

The FFN-only dynamic INT8 run also failed (decoder feature normalized RMS 0.44–0.70). This version quantizes only eligible FFN MatMul weights in groups of 64 along each output channel. Standard ONNX DequantizeLinear/Reshape/Transpose reconstruct FP32 weights; activation arithmetic stays FP32. No DynamicQuantizeLinear or integer MatMul is allowed in the decoder graph. The same 10% normalized RMS, exact top-token and READ/WRITE fixture gates apply. The encoder remains dynamic INT8. Packing compacts retained external data and records the decoder as `int8-weight-only`. IMPORTANT: ONNX Runtime may expand weights during session creation or inference; reduced file size is not guaranteed to reduce runtime memory or latency. This is a numerical/size experiment, not a native low-bit kernel or a proven Pixel-ready solution.

## Android four-stage synthetic integration test

CI now imports a hashed synthetic four-model ZIP including external INT8 weight data and standard DequantizeLinear, runs the actual Java streaming engine until it produces known waveform samples before EOF, checks that no additional audio is produced for unchanged tokens/final zero-duration comma, and checks that an invalid replacement ZIP leaves the previous package intact. These deterministic fixtures do not establish actual speech translation accuracy or real model memory consumption. The APK workflow now runs automatically only on Android-related changes and retries transient Maven HTTP 429/503 dependency failures. Real model conversion runs independently.

## Fine-group decoder and low-memory host experiment

Group-64 weight-only INT8 reduced logits RMS error to 0.049–0.072; the third case still failed decoder-feature RMS (0.139) and top-token agreement. Peak process RSS reached 6966 MiB after inference, despite only ~4.15 GB of external decoder weights. The next experiment uses group-16 scales for the same FFN projections, preserves all numerical gates, and disables host decoder constant folding/graph optimization, CPU arena retention, memory patterns, and weight prepacking. This can reduce persistent FP32 copies but may make inference slower; measure actual RSS and latency rather than assume savings. Host speech smoke uses the same decoder session configuration. Android configuration is not changed yet; if the host experiment passes, reproduce the memory configuration on Android and test actual device memory. Even a passing host accuracy gate does not establish lecture-length performance or sufficient memory for all four models.

## Independent acoustic-unit INT8 and native low-bit operator probes

Group-16 decoder weight-only INT8 preserved all three top tokens and READ/WRITE decisions but missed the feature gate in case 0 (0.108 > 0.10). Host peak RSS was 4878 MiB and timed decoder inference ~2 s, not suitable for a real-time target. The acoustic-unit experiment now has its own workflow: export the validated FP32 graph, reload the pinned reference model to create independent natural-English fixtures, and dynamically quantize constant matrix weights. Exact durations, exact unit argmax and <=0.10 logits normalized RMS are required. Reports record size and RSS; no experimental weights are packaged or substituted into the app. Android instrumentation probes the stock ORT MatMulNBits operator at 4 and 8 bits using known packed weights, requiring at least one supported precision and correct results for each loadable graph. This only tests operator availability and arithmetic on the emulator, not actual acceleration or Pixel performance. See https://onnxruntime.ai/docs/performance/kleidiai/ and https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html .

## Native packed decoder experiment and CI exit-code fix

The Android emulator probe produced correct results for both 4- and 8-bit MatMulNBits using the packaged ARM64 library. Acoustic-unit full dynamic INT8 shrank 1,064,032,094 bytes to 580,806,318 bytes but FAILED all five numerical fixtures (duration mismatches in four; unit mismatch in the remaining one). An implicit shell pipeline masked the Python failure. Explicit bash/pipefail and a PASS marker check now enforce that gate. The next unit experiment limits quantization to post-duration decoder FFNs. The decoder experiment directly emits packed UINT8 MatMulNBits with group-16 scales, trying symmetric and asymmetric weights, then a smaller output-projection-only subset. FP32 activations and the original numerical gates are retained. Native host experiments run independently and do not create or replace the Android model pack. Emulator arithmetic compatibility is not evidence of an optimized Pixel kernel or sustained real-time performance.

## Native decoder integration gate

Native asymmetric group-16 INT8 passed all three checkpoint fixtures with max logits NRMS 0.0643, feature NRMS 0.0945, exact top token and READ/WRITE decisions. Peak RSS was 3218 MiB and decoder inference 1.49–1.66 s on the host. This quantizes 48 FFN matrix operations, NOT the entire decoder. The model packaging workflow now reproduces the exact policy on ORT 1.30.0 and requires its PASS report; it retains the acoustic units in validated FP32 because the selective unit experiment could not identify eligible nodes. The integrated smoke test runs with ORT matching Android and logs per-session and per-step RSS. It is an experimental host smoke gate, not proof of meaningful translation, Android memory fit, or real-time performance. The current protective context limit remains.

## Android session alignment and acoustic-unit follow-up

Android decoder session options now match the passing native INT8 host experiment (CPU arena, memory-pattern retention and prepacking disabled; graph optimizations remain enabled). Per-model load time and process PSS are logged. The earlier acoustic-unit selection was wrong because the actual exported post-duration decoder has six self-attention layers with Q/K/V/output projections, not FFN MatMuls. The new independent experiment uses only those 24 native asymmetric INT8 projections, with a graph ancestry check proving none can affect durations. Unit argmax and numerical gates are unchanged. No acoustic-unit candidate is substituted into the package before passing. Android PSS and host RSS are different measures; neither synthetic tests nor the host run establish real device performance.

## Offline phone benchmark

The APK includes an optional benchmark for a locally selected 1–30 second mono signed 16-bit PCM 16 kHz WAV. It uses the actual four loaded model sessions, preserves transport frames and full-prefix policy, disables rollover, and reports load time, processing RTF, first generated audio, source consumed, output duration and waveform RMS, target count, and sampled process PSS. No microphone, playback or network is used. Copy the result with the app's Copy benchmark report button. This measures compute throughput (including session close), not live headphone latency or semantic accuracy. Sampled PSS can miss brief peaks; the 30-second diagnostic input limit is not a segmentation strategy or proof of lecture-length operation. No new models are needed beyond the verified import pack.


## Mobile optimization work (experimental)

The current Android prototype still re-encodes the complete accumulated source prefix and runs the complete target prefix on every update. The native asymmetric group-16 INT8 decoder remains the only accuracy-validated decoder; the native INT4 experiment was rejected because it changed top tokens and exceeded the 10% feature/logit error gate. The next decoder experiment exports an explicit self-attention KV cache: one graph accepts either a full prefix with empty caches or a single new token with populated caches. The cache is rebuilt from the accepted target prefix on each new audio source update, because decoder hidden states are source-conditioned; blindly reusing caches across changed encoder output would not be equivalent to Meta's reference agent. Cache outputs are tested against full-prefix PyTorch outputs at every token, then quantized to native INT8 and revalidated under ORT 1.30. This is not deployed unless all parity gates pass.

The live prototype keeps 320 ms capture frames but now runs full-prefix inference every 640 ms (first run after 1.92 s) to reduce repeated full-context encoder/decoder calls without dropping source frames. This trades some latency for fewer recomputations. Host speech diagnostics now separate encoder, decoder, UnitY2 and vocoder time, and count decoder calls and input tokens. FP16 candidates for the FP32 acoustic-unit and vocoder graphs are tested separately with the Android-matching ORT runtime; they are not substituted into a model pack unless durations/unit argmax or waveform tolerances pass. FP16 storage does not guarantee CPU acceleration.

## Incremental FP16 integration

Android accepts exactly one decoder variant: legacy INT8 or FP16 incremental plus cache metadata. Cache tensors remain in native ORT results and are reused within one source update; prefill is repeated when the source changes, and features for UnitY2 are accumulated per accepted token plus the synthesis comma. The Android fixture checks both decoder modes, including empty cache prefill, growth and release. The new packaging workflow starts from the validated FP16 acoustic package, verifies its hash inventory, replaces only the decoder and its unshared external weights, runs the real speech smoke test, and emits a new ZIP only after the incremental mode passes. This does not prove real-time speed, physical device compatibility, or semantic translation accuracy. Existing 60-second/100-token limits remain.

## FP16 memory profile experiment

The 3.72 GB incremental package passed the real speech waveform smoke (run 46002703236), but host peak RSS was 8375 MiB, including a large jump during FP16 decoder session construction. This can reflect CPU conversion/prepacking, and does not imply equivalent Android memory behavior. A new same-run isolated-process comparison tests default, no-retention (CPU arena, memory pattern and prepacking disabled), and basic graph optimization plus no-retention. Each profile uses the same 640 ms cadence and compares exact tokens and output duration to the same-run default, with logs and peak RSS/RTF. No automatic promotion, no Android settings changed until results are known. APK build retry limits were increased after Maven Central HTTP 429; no new dependency mirror added. The prior baseline cadence rerun also failed pre-EOF speech; this is retained as a failed result, not hidden or treated as a speed comparison.

## Cross-attention projection and ARM64 experiment

The pinned upstream OfflineWav2VecBertEncoderAgent calls encode_speech on the entire source. Shaw relative-position clipping is not proof of bounded receptive field. A drop-in exact encoder cache is therefore not assumed. Instead a graph partition experiment hoists only computations dependent on the encoder output and constants, preserving target-conditioned computations and external weights. The same runner compares the original and partitioned incremental decoder; report includes prefill READ/WRITE decisions, token agreement, source/fbank hashes, RSS, timings and architecture. Native ARM64 Linux is a closer instruction-set test, not Pixel or Android acceleration. No partitioned model is packaged or enabled in Android until parity passes. Previous memory profile runs failed pre-EOF speech and are not promoted. The legacy cadence benchmark now triggers only on its own workflow changes/manual dispatch, to avoid wasting runner resources on unrelated diagnostics.

## English-oriented vocabulary experiment

A 48,000-token budget selects Latin pieces in tokenizer ID order, reserves numeric/symbol tokens, preserves essential controls, byte fallback and an independent mathematics text seed. This is a heuristic, not an English language classifier or trained student model. The exact pinned upstream tokenizer supplies IDs; the real graph is rewritten only if all vocabulary-sized matrices have supported embedding/projection consumers. The compact projection restores original IDs with a finite excluded-token sentinel; UnitY2 and Java token interfaces stay unchanged. External tensors are compacted into a new weights file; baseline remains untouched. Both same-run speech tests must pass, with identical tokens, decisions, output duration and waveform NRMS <= 2%, before a candidate artifact is uploaded. No Android package promotion or semantic/lecture accuracy claim. Prior cross-attention split failed missing inferred boundary type on ARM and x64; remain on original graph.

## Full-vocabulary tied matrix INT8 experiment

The 48k pruning experiment (46008550493) saved 422,749,845 decoder bytes but changed accepted tokens, decisions and audio duration, and is rejected. The tied embedding/output matrix was confirmed by the actual graph conversion. Instead preserve all 256,102 original IDs and use shared UINT8 group-64 symmetric weights with FP32 scales. Gather only selected embedding rows and dequantize them; the output projection uses native MatMulNBits. FP16 activations outside this operation remain unchanged. Native x64 and ARM64 CPU unit tests and independent same-run speech comparisons must pass; no deployment or performance promise. CPU provider may still retain a packed copy. This is a numerical approximation, not lossless compression. Previous projection-cache split remains disabled.

## Native ARM64 results and cadence tuning

Run 46010132809 confirmed 4,437 MiB host peak RSS and RTF 3.52 for the unchanged incremental FP16 package on native ARM64. The paired x64 baseline reached 8,232 MiB and RTF 5.04; hardware differs. Tied INT8 converted successfully (245,873,320 bytes saved) but neither real graph loaded, due to a duplicate node name shared with the existing final projection Cast. The rewrite now allocates node names independently of tensor names; synthetic tests reproduce the collision pattern. Source checks use ORT 1.30/Python 3.12 rather than ORT 1.18, which only supports 4-bit MatMulNBits. A separate native ARM64 same-run cadence benchmark compares 640/960/1280/1600 ms and thread counts. No audio is dropped, no independent chunks are introduced, but different update times can alter decisions and spoken output. Strict matching and waveform comparisons are reported; no Android rollout or semantic accuracy claim. Both total elapsed RTF and processing-only RTF are reported. More complete input and device testing remains necessary.

## On-device execution-provider diagnostics

The 48k vocabulary candidate failed parity and is rejected. The full-vocabulary tied UINT8/MatMulNBits experiment is also rejected: it was slower (x64 decoder 85.5s vs 22.7s FP16) and failed ARM64 token parity. The native ARM64 cadence test on a 4-vCPU Neoverse-N2 runner found 640ms/4 threads exact fixture parity at processing RTF 3.31; 960/1280/1600ms reduced processing RTF to 2.74/2.16/1.95 but changed token sequence and duration, so these are not deployable settings without semantic evaluation. Two threads were much slower (RTF 6.02).

The upstream OfflineWav2VecBertEncoderAgent explicitly recomputes `encode_speech` over the entire accumulated source on every update. Shaw relative-position clipping is not a bounded attention window; do not infer that a 64-frame cache or prefix truncation is exact. A true encoder cache requires an architecture-aware causal/chunked encoder and training/quality validation. This is the next major algorithmic project.

The diagnostic APK now optionally requests XNNPACK for FP32 UnitY2/vocoder stages and NNAPI for the INT8 speech encoder, while retaining CPU fallback. It adds a real 2-D MatMul XNNPACK instrumentation smoke test. These are opt-in and off by default. XNNPACK's Android package includes the provider, but its supported operators and actual partitioning are model-dependent; NNAPI is device-dependent and deprecated in Android 15. No claim of Pixel 9a acceleration until the local WAV benchmark runs on the physical phone. Benchmark reports identify requested providers. The branch APK remains a separate experimental test application, not the production RTranslator APK.


## Real-weight Vulkan projection parity experiment

A dedicated experimental APK can embed one FP16 MatMul weight matrix and a representative activation captured from the actual Mandarin encoder plus incremental decoder source-projection graph. The APK runs an isolated FP32 ONNX Runtime CPU reference on-device, then runs the same MxK by KxN projection through a native Vulkan fused-4 shader using the exact FP16 model weights. Every output is compared; the report includes CPU ORT time, GPU timestamp, queue/fence latency, host fixture-copy time and Vulkan setup time. This is a single real model projection experiment, not full decoder acceleration, not an end-to-end translation benchmark, and not a validated production provider. The extracted Meta weights are CC BY-NC 4.0 and noncommercial only; the attribution/license file is included in the APK assets.


The real-projection fixture keeps the weight matrix in its original ONNX MatMul layout. The Android native path measures the FP16 transpose/packing into output-major Vulkan storage separately from activation copy and pipeline/device setup; this makes the one-time packing cost visible rather than hiding it in CI fixture generation.
