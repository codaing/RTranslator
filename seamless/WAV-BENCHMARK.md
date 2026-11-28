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
