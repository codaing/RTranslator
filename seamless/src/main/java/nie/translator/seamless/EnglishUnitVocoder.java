package nie.translator.seamless;

import java.nio.LongBuffer;
import java.util.Collections;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

/**
 * Experimental SeamlessStreaming acoustic-unit -> English waveform component.
 * Requires a parity-validated vocoder.onnx from seamless-port/export_components.py.
 * Not a text-to-speech engine; the caller must supply UnitY2 acoustic unit IDs.
 * Not connected to the Android UI until all upstream stages are ported.
 */
public final class EnglishUnitVocoder implements AutoCloseable {
    public static final int SAMPLE_RATE = 16000;
    private final OrtEnvironment environment;
    private final OrtSession session;

    public EnglishUnitVocoder(String modelPath) throws OrtException {
        this(modelPath,false);
    }
    EnglishUnitVocoder(String modelPath,boolean quietCpu) throws OrtException {
        environment = OrtEnvironment.getEnvironment();
        try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
            if(quietCpu)CpuTuning.quiet(options,false);
            session = environment.createSession(modelPath, options);
        }
    }

    /** Decode a single contiguous batch of units, preserving their order. */
    public synchronized float[] decode(long[] units) throws OrtException {
        if (units == null || units.length == 0) return new float[0];
        for (long unit : units) {
            if (unit < 0 || unit >= 10000) {
                throw new IllegalArgumentException("Invalid acoustic unit: " + unit);
            }
        }
        try (OnnxTensor input = OnnxTensor.createTensor(environment,
                LongBuffer.wrap(units), new long[]{1, units.length});
             OrtSession.Result result = session.run(Collections.singletonMap("units", input))) {
            Object value = result.get("waveform").orElseThrow(
                    () -> new IllegalStateException("Missing waveform output")).getValue();
            if (!(value instanceof float[][][])) {
                throw new IllegalStateException("Expected [1,1,samples] waveform");
            }
            float[][][] wave = (float[][][]) value;
            if (wave.length != 1 || wave[0].length != 1) {
                throw new IllegalStateException("Unexpected waveform shape");
            }
            float[] samples = wave[0][0];
            for (float sample : samples) {
                if (!Float.isFinite(sample)) {
                    throw new IllegalStateException("Non-finite waveform");
                }
            }
            return samples;
        }
    }

    @Override
    public synchronized void close() throws OrtException {
        session.close();
    }
}
