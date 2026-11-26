# SeamlessStreaming Android port — phase 1 (not an Android translator yet)

Goal: continuous Mandarin lecture audio to English speech on Pixel 9a, with no Android TextToSpeech, no Whisper/HY-MT cascade, no cloud service, and no fixed utterance cutoff.

This experimental branch deliberately leaves the functioning RTranslator APK alone. It contains an offline reference runner for the actual Meta SeamlessStreaming S2ST pipeline, an ONNX component-export experiment with numerical checks, and an Android English unit-vocoder adapter. It does not yet wire the model into the Android UI. Do not distribute an APK from this branch as a SeamlessStreaming translator.

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

Remaining work, in order:
1. Validate encoder, decoder and vocoder exports on real checkpoints. Verify dynamic shapes, numerical parity and output audio.
2. Port UnitY2 text-token to character mapping, hard upsampling/duration prediction and acoustic-unit decoder. Its Python loops and data-dependent lengths are not safe to simply trace. Export the tensor-only stages; validate acoustic units and alignment against the reference.
3. Port the filterbank extractor (25 ms windows/10 ms hop, 80 mel bins), EMMA read/write scheduling and streaming state to Android. Preserve context; define a semantic/context-limit reset strategy for long lectures rather than fixed five-second cuts.
4. Quantize only after FP32 parity; test memory and real-time factor on Pixel 9a, including sustained thermal load. Require output before source EOF, no missing/duplicated words, and real-time factor below one on sustained Mandarin input.
5. Replace the UI/model downloader with a single Mandarin-to-English mic/headset screen only once the above runs locally; then remove legacy ASR/translation/TTS on this branch.

No model weights are committed. No backend server, network inference, or fallback TTS is included in this work. No successful model export or Android performance claim is made yet.
