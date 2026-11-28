# Production real-time track

## Decision

Prioritize the existing :app RTranslator pipeline for the first production release: Whisper ONNX speech recognition -> the app's selected text-translation engine -> Android TTS. Keep :seamless as a separate research application until its complete speech pipeline is demonstrably real-time and semantically reliable.

This is a pragmatic no-fine-tuning path. The SeamlessStreaming prototype is not currently a production candidate: the supplied Pixel 9a run measured processing RTF 11.091, first generated audio at 8.55 s, and sampled process PSS near 4 GiB. Encoder plus decoder prefill accounted for 68.1% of measured processing time. The k448/k512 FFN numbers are isolated subgraph results; they do not establish complete translation throughput.

## Low-latency profile added to the existing app

- Speech-recognition beam defaults to 1, configurable from Settings -> Advanced -> Speech recognition beam (1-4). The old translation beam setting remains separate.
- Default end-of-speech timeout is reduced from 1.3 s to 0.7 s. It remains configurable in Settings, so users can increase it if pauses split their phrases.
- Voice translation already uses translation beam 1.
- Whisper and translation ONNX sessions remain resident; this change does not reload models per utterance.
- Structured SpeechPipeline log events report voice segment duration, ASR elapsed time/audio duration/beam, MT elapsed time/input size/beam/model mode, and TTS start/end duration.

The fast defaults trade some recognition robustness and pause tolerance for responsiveness. They are reversible through the settings sliders. They are a starting profile, not yet a verified real-time claim.

## Physical-device acceptance

Run the same 30-60 second speech fixture at least five times on Pixel 9a, after the device is cool. Capture SpeechPipeline, performance, recognizer, result, and tts_engine log tags. Compare beam 1 vs 2 and endpoint timeout 0.7 vs 1.0 seconds, with fixed source/target languages (avoid auto language detection during the latency baseline).

Do not declare real-time until:
- sustained end-to-end processing RTF < 1.0 on the target phone;
- median time from speech endpoint to first translated audio is measured and acceptable;
- no queue growth or dropped speech over a 10-minute session;
- transcript/translation quality is reviewed on held-out conversational speech;
- microphone, speaker/headset routing, interruptions, screen-off foreground behavior, and recovery are tested.

Do not promote experimental TPU FFN or Seamless models into the live app based only on microbenchmarks. Any future acceleration must pass full-model output parity and end-to-end speech evaluation.
