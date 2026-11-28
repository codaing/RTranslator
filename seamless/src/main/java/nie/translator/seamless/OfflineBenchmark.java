package nie.translator.seamless;

import java.io.*;
import java.security.MessageDigest;
import java.util.Arrays;
import java.util.Locale;
import java.util.concurrent.CancellationException;
import java.util.function.BooleanSupplier;

import android.os.Debug;
import android.os.SystemClock;

/**
 * Offline WAV benchmark. Uses the same 320 ms source frames as the streaming path.
 * Primary processing RTF is measured around engine.push calls, including final flush,
 * but excluding model load, PSS sampling, progress reporting and session close.
 */
final class OfflineBenchmark {
    static String run(File dir, InputStream in, BooleanSupplier cancelled, LiveAudio.Status status,
                      boolean xnnpackFp32Stages, boolean nnapiEncoder, boolean quietCpu) throws Exception {
        return run(dir, in, cancelled, status, xnnpackFp32Stages, nnapiEncoder, quietCpu,
                GpuTuning.CPU, "selected.wav");
    }

    static String run(File dir, InputStream in, BooleanSupplier cancelled, LiveAudio.Status status,
                      boolean xnnpackFp32Stages, boolean nnapiEncoder, boolean quietCpu,
                      GpuTuning gpu) throws Exception {
        return run(dir, in, cancelled, status, xnnpackFp32Stages, nnapiEncoder, quietCpu,
                gpu, "selected.wav");
    }

    static String run(File dir, InputStream in, BooleanSupplier cancelled, LiveAudio.Status status,
                      boolean xnnpackFp32Stages, boolean nnapiEncoder, boolean quietCpu,
                      GpuTuning gpu, String wavName) throws Exception {
        return run(dir,in,cancelled,status,xnnpackFp32Stages,nnapiEncoder,quietCpu,gpu,wavName,StreamingEngine.UPDATE_INTERVAL_FRAMES);
    }

    static String run(File dir, InputStream in, BooleanSupplier cancelled, LiveAudio.Status status,
                      boolean xnnpackFp32Stages, boolean nnapiEncoder, boolean quietCpu,
                      GpuTuning gpu, String wavName, int updateFrames) throws Exception {
        return run(dir,in,cancelled,status,xnnpackFp32Stages,nnapiEncoder,quietCpu,gpu,wavName,updateFrames,false);
    }

    static String run(File dir, InputStream in, BooleanSupplier cancelled, LiveAudio.Status status,
                      boolean xnnpackFp32Stages, boolean nnapiEncoder, boolean quietCpu,
                      GpuTuning gpu, String wavName, int updateFrames, boolean traceProviders) throws Exception {
        short[] samples = BenchmarkWav.read(in);
        if (cancelled.getAsBoolean()) throw new CancellationException("Cancelled before model load");

        String audioHash = sha256Pcm16(samples);
        String modelHash = sha256File(new File(dir, "sha256.json"));
        status.report("Benchmark: loading real model sessions…");

        long loadStart = SystemClock.elapsedRealtime();
        long loadMs;
        long computeStartMs;
        long computeStartNanos;
        long computeWallNanos;
        long closeStartNanos;
        long closeNanos;
        final long[] generated = {0}, first = {-1}, firstSource = {-1}, peak = {Debug.getPss()};
        final long[] consumed = {0};
        final boolean[] eof = {false}, beforeEof = {false};
        final double[] energy = {0};
        int tokens;
        boolean finalEos;
        String decoderMode, stageReport, tokenFingerprint, providerReport;
        ProviderProfile profile=traceProviders?new ProviderProfile(dir.getParentFile()):null;
        long pushNanos = 0, pssNanos = 0;

        try (StreamingEngine engine = new StreamingEngine(
                dir, false, xnnpackFp32Stages, nnapiEncoder, quietCpu, gpu, updateFrames, profile)) {
            loadMs = SystemClock.elapsedRealtime() - loadStart;
            peak[0] = Math.max(peak[0], Debug.getPss());
            if (cancelled.getAsBoolean()) throw new CancellationException("Cancelled after model load");

            computeStartMs = SystemClock.elapsedRealtime();
            computeStartNanos = System.nanoTime();
            StreamingEngine.Sink sink = pcm -> {
                generated[0] += pcm.length;
                for (float f : pcm) energy[0] += (double) f * f;
                if (first[0] < 0) {
                    first[0] = SystemClock.elapsedRealtime() - computeStartMs;
                    firstSource[0] = consumed[0];
                    beforeEof[0] = !eof[0];
                }
            };

            for (int i = 0; i < samples.length; i += 5120) {
                if (cancelled.getAsBoolean()) throw new CancellationException("Cancelled");
                int end = Math.min(samples.length, i + 5120);
                consumed[0] = end;
                short[] frame = Arrays.copyOfRange(samples, i, end);
                long pushStart = System.nanoTime();
                engine.push(frame, false, sink);
                pushNanos += System.nanoTime() - pushStart;

                long pssStart = System.nanoTime();
                peak[0] = Math.max(peak[0], Debug.getPss());
                pssNanos += System.nanoTime() - pssStart;
                status.report(String.format(Locale.US,
                        "Benchmark: %.2f / %.2f s source | generated %.2f s | sampled PSS %.0f MiB",
                        end / 16000.0, samples.length / 16000.0,
                        generated[0] / 16000.0, peak[0] / 1024.0));
            }

            if (cancelled.getAsBoolean()) throw new CancellationException("Cancelled before final flush");
            eof[0] = true;
            long flushStart = System.nanoTime();
            engine.push(new short[0], true, sink);
            pushNanos += System.nanoTime() - flushStart;

            long pssStart = System.nanoTime();
            peak[0] = Math.max(peak[0], Debug.getPss());
            pssNanos += System.nanoTime() - pssStart;

            tokens = engine.targetTokens();
            finalEos = engine.finalEos();
            decoderMode = engine.decoderMode();
            tokenFingerprint = engine.tokenFingerprint();
            stageReport = engine.stats.report();
            computeWallNanos = System.nanoTime() - computeStartNanos;
            providerReport=engine.providerReport();
            closeStartNanos = System.nanoTime();
        }

        closeNanos = System.nanoTime() - closeStartNanos;
        double duration = samples.length / 16000.0;
        double processingSeconds = pushNanos / 1e9;
        double wallSeconds = computeWallNanos / 1e9;
        double processingRtf = processingSeconds / duration;
        double wallRtf = wallSeconds / duration;

        return String.format(Locale.US,
                "OFFLINE WAV RTF BENCHMARK\n" +
                "Device: %s | Android %s (SDK %d)\n" +
                "Runtime: ONNX Runtime Android 1.30.0 | CPU intra-op 4 / inter-op 1\n" +
                "Decoder: %s | Requested provider profile: "+gpu+"\n" +
                "WAV: %s\n" +
                "Input PCM SHA-256: %s\n" +
                "Model manifest SHA-256: %s\n" +
                "Input audio: %.2f s | generated audio: %.2f s\n" +
                "Model/session load: %.2f s (excluded from processing RTF)\n" +
                "Engine processing: %.2f s (push calls + final flush)\n" +
                "PROCESSING RTF: %.3f (primary comparison; <1.0 means faster than audio duration)\n" +
                "Harness wall RTF: %.3f (includes progress and PSS sampling; secondary)\n" +
                "Session close: %.2f s (excluded from processing RTF)\n" +
                "First generated audio: %s | source consumed: %s | before final flush: %s\n" +
                "Sampled peak PSS: %.0f MiB | target tokens: %d\n" +
                "Waveform RMS: %.6f\n" +
                "No microphone, playback, network or TTS used. RTF does not measure translation accuracy.",
                android.os.Build.MODEL, android.os.Build.VERSION.RELEASE, android.os.Build.VERSION.SDK_INT,
                decoderMode, wavName, audioHash, modelHash, duration, generated[0] / 16000.0,
                loadMs / 1000.0, processingSeconds, processingRtf, wallRtf, closeNanos / 1e9,
                first[0] < 0 ? "none" : String.format(Locale.US, "%.2f s", first[0] / 1000.0),
                firstSource[0] < 0 ? "none" : String.format(Locale.US, "%.2f s", firstSource[0] / 16000.0),
                beforeEof[0] ? "yes" : "no", peak[0] / 1024.0, tokens,
                generated[0] > 0 ? Math.sqrt(energy[0] / generated[0]) : 0)
                + "\nBenchmark schedule: " + (updateFrames==0?"EOF-only (not streaming)":String.format(Locale.US,"%.2f s nominal update interval",updateFrames/100.0))
                + "\nQuiet CPU thread pools: " + quietCpu
                + "\nFinal decoder EOS reached: " + finalEos
                + "\nOutput completeness screen (EOS and nonempty audio): " + (finalEos&&generated[0]>0?"PASS":"FAIL")
                + "\nProvider profiling enabled: " + traceProviders
                + (traceProviders?" (diagnostic timings; profiling adds overhead)":"")
                + "\n" + providerReport
                + "\nRequested experimental providers: " + gpu
                + "\nToken SHA-256: " + tokenFingerprint
                + "\nPSS sampling overhead: " + String.format(Locale.US, "%.3f s", pssNanos / 1e9)
                + "\n\n" + stageReport
                + String.format(Locale.US,
                    "Engine push time: %.3f s | input items: PCM samples / feature frames / decoder tokens / characters / acoustic units.",
                    pushNanos / 1e9);
    }

    private static String sha256Pcm16(short[] samples) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        for (short sample : samples) {
            digest.update((byte) (sample & 0xff));
            digest.update((byte) ((sample >>> 8) & 0xff));
        }
        return hex(digest.digest());
    }

    private static String sha256File(File file) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        try (InputStream in = new BufferedInputStream(new FileInputStream(file))) {
            byte[] buffer = new byte[1024 * 1024];
            int count;
            while ((count = in.read(buffer)) != -1) digest.update(buffer, 0, count);
        }
        return hex(digest.digest());
    }

    private static String hex(byte[] bytes) {
        StringBuilder result = new StringBuilder(bytes.length * 2);
        for (byte value : bytes) result.append(String.format(Locale.US, "%02x", value & 0xff));
        return result.toString();
    }
}
