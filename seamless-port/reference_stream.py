#!/usr/bin/env python3
"""Replay continuous PCM into Meta's actual SeamlessStreaming agent, no TTS API."""
import argparse
import array
import sys
import time
import wave


def pcm_frames(reader, size):
    """One-chunk lookahead: exactly one EOF marker, no synthetic utterance cuts."""
    current = reader.readframes(size)
    if not current:
        raise ValueError("Input WAV is empty")
    while current:
        following = reader.readframes(size)
        samples = array.array("h")
        samples.frombytes(current)
        if sys.byteorder != "little":
            samples.byteswap()
        yield [sample / 32768.0 for sample in samples], not following
        current = following


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--frame-ms", type=int, default=320)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not 40 <= args.frame_ms <= 1000 or not 0 <= args.threshold <= 1:
        parser.error("frame-ms must be 40..1000 and threshold 0..1")
    if args.source == args.output:
        parser.error("source and output must differ")
    # Import lazily so PCM handling can be tested without downloading any models.
    import numpy as np
    from simuleval.data.segments import SpeechSegment
    from seamless_communication.streaming.agents.seamless_streaming_s2st import (
        SeamlessStreamingS2STAgent,
    )
    # Defaults from upstream streaming evaluation (fixed English output).
    config = argparse.Namespace(
        task="s2st", device=args.device, dtype=args.dtype, fp16=False,
        unity_model_name="seamless_streaming_unity",
        monotonic_decoder_model_name="seamless_streaming_monotonic_decoder",
        sample_rate=16000, tgt_lang="eng", shift_size=10, window_size=25,
        feature_dim=80, denormalize=True, min_starting_wait_w2vbert=192,
        min_starting_wait=1, decision_threshold=args.threshold,
        decision_method="min", p_choose_start_layer=0, block_ngrams=False,
        no_early_stop=True, max_len_a=0, max_len_b=100,
        max_consecutive_write=50, min_unit_chunk_size=50, d_factor=1.0,
        vocoder_name="vocoder_v2", vocoder_speaker_id=-1,
    )
    with wave.open(args.source, "rb") as source:
        if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (1, 2, 16000):
            raise ValueError("Input must be mono signed 16-bit PCM at 16000 Hz")
        agent = SeamlessStreamingS2STAgent(config)
        agent.reset()
        with wave.open(args.output, "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(16000)
            start = time.monotonic()
            first_output = None
            input_samples = 0
            output_samples = 0
            output_before_eof = False

            def consume(segment, eof):
                nonlocal first_output, output_samples, output_before_eof
                if segment.is_empty or not isinstance(segment, SpeechSegment):
                    return
                if segment.sample_rate != 16000:
                    raise ValueError("Unexpected output sample rate")
                audio = np.asarray(segment.content, dtype=np.float32).reshape(-1)
                if not audio.size:
                    return
                if not np.isfinite(audio).all():
                    raise ValueError("Non-finite waveform")
                if first_output is None:
                    first_output = time.monotonic() - start
                    print(f"First audio: {first_output:.3f}s; source consumed: {input_samples / 16000:.3f}s", flush=True)
                output_before_eof |= not eof
                pcm = np.clip(audio * 32768, -32768, 32767).astype("<i2")
                out.writeframes(pcm.tobytes())
                output_samples += pcm.size

            result = None
            for samples, eof in pcm_frames(source, args.frame_ms * 16):
                input_samples += len(samples)
                if args.realtime:
                    time.sleep(max(0, start + input_samples / 16000 - time.monotonic()))
                result = agent.pushpop(SpeechSegment(
                    content=samples, sample_rate=16000, tgt_lang="eng", finished=eof))
                consume(result, eof)
            # Drain after source EOF, bounded so a bad model cannot loop forever.
            for _ in range(256):
                if result.finished:
                    break
                result = agent.pop()
                consume(result, True)
            else:
                raise RuntimeError("Model did not finish after EOF")
            elapsed = time.monotonic() - start
    if not output_samples:
        raise RuntimeError("Model produced no speech")
    print(f"Input: {input_samples/16000:.2f}s; output: {output_samples/16000:.2f}s")
    print(f"Wall-clock / input duration: {elapsed/(input_samples/16000):.3f}")
    print(f"Output before EOF: {output_before_eof}")
    if not output_before_eof:
        raise RuntimeError("No speech before EOF; streaming check failed")


if __name__ == "__main__":
    main()
