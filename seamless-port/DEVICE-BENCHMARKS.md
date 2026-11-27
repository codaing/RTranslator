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
