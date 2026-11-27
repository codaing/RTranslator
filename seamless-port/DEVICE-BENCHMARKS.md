# Physical device benchmarks — 2026-11-28

User-provided results from identical 13.34-second Mandarin WAV, incremental FP16 decoder and Android 17. All generated 8.40 s of waveform, 47 target tokens, first output before flush. This is one run per configuration; temperature, scheduling, memory pressure and load conditions are uncontrolled. Neither token IDs nor audio content were compared. Provider requested does not establish hardware assignment.

| Device | NNAPI requested | XNNPACK units requested | Load s | Processing s | RTF | First output s | PSS MiB | RMS |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Pixel 9a | no | no | 15.76 | 164.91 | 12.360 | 12.45 | 3792 | 0.079179 |
| Pixel 9a | no | yes | 24.51 | 282.43 | 21.169 | 9.99 | 3603 | 0.078976 |
| Pixel 9a | yes | no | 42.71 | 452.37 | 33.906 | 36.60 | 3813 | 0.079179 |
| Pixel 9a | yes | yes | 20.89 | 263.01 | 19.713 | 11.71 | 3946 | 0.078976 |
| Pixel 7 | no | no | 71.39 | 158.14 | 11.853 | 30.72 | 3949 | 0.079179 |
| Pixel 7 | yes | yes | 16.07 | 155.04 | 11.620 | 6.68 | 3990 | 0.078976 |
| Pixel 7 | yes | no | 20.61 | 148.98 | 11.166 | 6.63 | 4506 | 0.079179 |
| Pixel 7 | no | yes | 20.02 | 160.39 | 12.022 | 6.41 | 3986 | 0.078976 |

First output at 2.88 s of source consumed in every run. Processing includes session close; excludes load. PSS sampled, not true peak RSS. CPU only remains default; NNAPI and XNNPACK are not promoted. The old XNNPACK label was inaccurate: vocoder used its own CPU session. Current diagnostics correct label and preserve provider behavior. No model changes.

New instrumentation times filterbank, full-prefix encoder, decoder prefill, incremental steps, synthesis comma, unit decoder and vocoder separately, with call and input item counts. Reports push, sampled PSS overhead and close time. Unit ORT results are released before vocoder inference. Quiet CPU option requests 4 intra-op/1 inter-op threads and disables idle spinning in all four CPU sessions (XNNPACK keeps its own pool); disabled by default. This is a combined thread-pool experiment, not isolated proof of spinning effect. Synthetic fixtures gate waveform output; no real-time claim. Compare just two CPU-only runs of the same recording, after cooling; no need to repeat eight provider tests. Existing model pack is unchanged.

## Profiled Pixel 9a follow-up

Same 13.34 s recording; providers off. Normal vs quiet CPU: load 13.91/15.63 s; processing 147.98/191.38 s; RTF 11.091/14.344; sampled PSS 3999/3969 MiB; first audio 8.55/7.78 s at source 2.88 s. Both 47 tokens, 8.40 s output, RMS 0.079179. One run each; not a controlled thermal experiment or proof of waveform equality.

| Stage | Normal s | Quiet s | Calls | Items |
|---|---:|---:|---:|---:|
| Filterbank | .139 | .187 | 43 | 213472 |
| Encoder | 63.412 | 79.884 | 19 | 15120 |
| Decoder prefill | 37.314 | 41.437 | 19 | 570 |
| Decoder token steps | 14.084 | 22.974 | 47 | 47 |
| Synthesis comma | 3.249 | 5.459 | 12 | 12 |
| Acoustic units | 20.756 | 29.829 | 12 | 1251 |
| Vocoder | 1.572 | 2.339 | 6 | 420 |
| PSS sampling | 5.799 | 7.170 | | |
| Close | 1.560 | 2.001 | | |

Quiet CPU remains off. Encoder + prefill 68.1% of normal wall time. Even removing their entire time would leave 47.25 s (RTF ~3.54), so caching alone cannot meet real time. Retry source projection partition (previous missing shape /text_decoder/layers.0/encoder_decoder_attn/Transpose_1_output_0) with small external shape tensors materialized in metadata copy and source-only symbolic inference fallback. Native ARM64 same-run comparison now requires identical accepted tokens, decisions, duration, waveform NRMS <=2%, and improved processing RTF. No Android integration before passing. No independent TTS, no fixed utterance cuts. Existing package unchanged.

## Native ARM64 projection cache (not a phone benchmark)

Run 46082615070, commit 47fe2eb: 144 source-only MatMul/Gemm operations, 72 projection boundaries. Processing RTF 3.228079 -> 2.918371 (9.6% lower); accepted tokens, prefill decisions, duration identical; PCM NRMS 0.0. One synthetic Mandarin speech fixture, one pair of measurements. Candidate peak RSS 4224.85 MiB. Do not infer Android speed or general translation quality. Optional Android projected model pack shares external decoder weights and keeps native projection results alive only within each source update; legacy packages remain supported. Android fixture and separate host-gated package must pass before physical device testing.

Second native ARM64 source projection run 46083582838 passed: processing RTF 3.122283 -> 2.919721 (6.5% lower); peak RSS 4443.30 -> 4201.98 MiB; exact token, decision, duration and PCM parity. Artifact 15238212593 (3.73 GB ZIP) and APK run 46083594265 artifact 15238586622 (all eight synthetic instrumentation tests passed). Physical-device speed is unverified.

Next encoder investigation: measure actual prefix drift and simulate aligned overlapping windows (3.84, 7.68, 11.52 seconds, 1.28 s guard), retaining old encoder states instead of discarding speech. This is an approximation diagnostic, not a true layer KV cache or a quality gate. No Android changes or candidate model packaging from this probe. Repeated pinned speech is not a mathematics lecture dataset.
