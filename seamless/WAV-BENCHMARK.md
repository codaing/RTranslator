# Standalone WAV RTF benchmark

This is a separate Android application (package: nie.translator.seamless), not the RTranslator app. It exists to compare the experimental Seamless model and runtime changes on the same Pixel 9a and the same input recording.

## Phone workflow

1. Install the Seamless WAV RTF Benchmark APK.
2. Tap Import model package (.zip) once and select the verified model-pack ZIP. The app verifies the package SHA-256 manifest before activation.
3. Tap Choose WAV file and select the fixture.
4. Tap Run CPU RTF benchmark.
5. Tap Copy report and save/send the report.

## WAV fixture requirements

- RIFF/WAVE
- Uncompressed PCM
- Mono
- 16,000 Hz
- Signed 16-bit little-endian samples
- 1–30 seconds

No resampling, channel mixing, microphone capture, playback, network or TTS is involved. Convert a source recording on a computer if necessary:

    ffmpeg -i source-audio -ac 1 -ar 16000 -c:a pcm_s16le fixture.wav

## How to compare runs

Use the same WAV and model package. Reports include the PCM SHA-256 and model-manifest SHA-256 so that mismatched inputs are visible.

PROCESSING RTF is the primary metric. It is the summed wall time of the model engine's source-frame push calls, including the final flush, divided by source audio duration. Model/session loading, PSS sampling, progress reporting and session close are excluded. This aligns with the live prototype's processing RTF definition.

The report also includes a secondary harness wall RTF, model load and session-close time, first generated audio, output duration, sampled peak PSS, target-token count, token fingerprint and per-stage timings.

RTF below 1.0 means the measured model pipeline processes faster than the duration of the input audio. This is a throughput measurement, not proof of semantic translation accuracy, live microphone latency, stable long-form streaming or production readiness.

## Baseline settings

- CPU execution provider
- ONNX Runtime Android 1.30.0
- 4 intra-op threads
- 1 inter-op thread
- 320 ms source input frames
- Adaptive context rollover disabled

## Pixel 9a baseline and fast schedules (2026-11-28)
Input 1.wav PCM SHA-256 c97190975ee8d73932af558bbcaf7e827dcbca39cf65acbb295a06993eef26c8, incremental FP16 manifest 7e5236511f351567afe4fb9bcf1f8a0b1484879999bbd1e40b3c513da05cb61a. 13.34 s input, 405.541 s processing (RTF 30.396), 8.40 s generated, 47 tokens, peak sampled PSS 3659 MiB. Encoder 183.464 s/19 calls; prefill 117.274 s/19 calls; token steps 44.077 s; units 47.401 s. Target 0.8 requires roughly 38x improvement. Baseline mode remains unchanged. Fast mode uses 256 feature frames between updates; EOF-only mode accumulates the same 320 ms input frames but calls the encoder and synthesis only at final flush. EOF-only is a throughput diagnostic, not streaming. No model weights changed. The independent quiet CPU checkbox disables idle ORT spinning (also sets vocoder to four threads, unlike original default); timings and generated tokens can differ. Reports include schedule, quiet mode and final EOS/nonempty audio screen. Do not promote a mode that loses speech. Test projected FP16 with same input, then EOF-only and fast schedules; compare actual generated audio and token fingerprint before interpreting speed. Model weights remain noncommercial under CC BY-NC 4.0.
