package nie.translator.seamless;

import android.content.Context;
import android.os.Build;
import android.util.Log;

import org.json.JSONArray;
import org.json.JSONObject;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.nnapi.NnApiDelegate;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

/**
 * Experimental in-process mixed FFN benchmark.
 *
 * Keeps both TFLite interpreters alive in one Android process:
 * compact FP32 M=1 on XNNPACK, then INT8 M=31 on NNAPI google-edgetpu.
 * The timed path includes row slicing/copy, INT8 input quantization,
 * TPU invocation, output dequantization, and row reassembly.
 */
final class MixedFfnBenchmark {
    private static final int ROWS = 32;
    private static final int TAIL_ROWS = 31;
    private static final int CHANNELS = 1024;
    private static final int GROUP = 64;
    private static final int GROUPS = CHANNELS / GROUP;
    private static final String COMPACT_MODEL = "ffn-m1-compact-k512-fp32.tflite";
    private static final String TAIL_MODEL = "ffn-m31-m31-prefill-tail-g64-int8.tflite";

    private MixedFfnBenchmark() {}

    static String run(File dir, int warmups, int iterations) throws Exception {
        if (warmups < 1 || iterations < 20) {
            throw new IllegalArgumentException("need at least one warmup and 20 timed iterations");
        }
        File compactFile = new File(dir, COMPACT_MODEL);
        File tailFile = new File(dir, TAIL_MODEL);
        File inputFile = new File(dir, "mixed-full32-input.f32");
        File referenceFile = new File(dir, "mixed-full32-reference.f32");
        File manifestFile = new File(dir, "tail-manifest.json");
        for (File f : new File[]{compactFile, tailFile, inputFile, referenceFile, manifestFile}) {
            if (!f.isFile() || f.length() == 0) throw new IllegalStateException("missing fixture: " + f);
        }

        float[] input = readFloat32(inputFile, ROWS * CHANNELS);
        float[] reference = readFloat32(referenceFile, ROWS * CHANNELS);
        JSONObject tailMeta = findTailMeta(manifestFile);
        int[] groupToTensor = readGroupMapping(tailMeta);
        JSONArray outputQuantizations = tailMeta.getJSONArray("output_quantizations");
        JSONArray inputQuantization = tailMeta.getJSONArray("input_quantization");
        if (tailMeta.getInt("rows") != TAIL_ROWS || tailMeta.getInt("group_size") != GROUP
                || tailMeta.getInt("output_groups") != GROUPS) {
            throw new IllegalStateException("tail manifest is not M=31/group64");
        }

        long initStart = System.nanoTime();
        MappedByteBuffer compactBuffer = map(compactFile);
        MappedByteBuffer tailBuffer = map(tailFile);

        Interpreter compact = null;
        Interpreter tail = null;
        Interpreter tailCpu = null;
        NnApiDelegate nnapi = null;
        try {
            Interpreter.Options compactOptions = new Interpreter.Options()
                    .setNumThreads(4)
                    .setUseXNNPACK(true);
            compact = new Interpreter(compactBuffer, compactOptions);
            Tensor compactInputTensor = compact.getInputTensor(0);
            Tensor compactOutputTensor = compact.getOutputTensor(0);
            requireTensor(compactInputTensor, DataType.FLOAT32, new int[]{1, 1, 1, CHANNELS}, "compact input");
            requireTensor(compactOutputTensor, DataType.FLOAT32, new int[]{1, 1, 1, CHANNELS}, "compact output");

            // One CPU-only tail invocation establishes byte-exact parity with the TPU delegate.
            Interpreter.Options tailCpuOptions = new Interpreter.Options()
                    .setNumThreads(4)
                    .setUseXNNPACK(true);
            tailCpu = new Interpreter(tailBuffer, tailCpuOptions);
            TailBuffers cpuTailBuffers = allocateTailBuffers(tailCpu);
            validateTailModel(tailCpu, tailMeta, inputQuantization, outputQuantizations, groupToTensor);
            ByteBuffer quantizedTailInput = allocateInput(tailCpu.getInputTensor(0).numBytes());
            fillQuantizedTail(input, quantizedTailInput,
                    tailCpu.getInputTensor(0).quantizationParams().getScale(),
                    tailCpu.getInputTensor(0).quantizationParams().getZeroPoint());
            runTail(tailCpu, quantizedTailInput, cpuTailBuffers);
            byte[][] cpuTailBytes = copyOutputBytes(cpuTailBuffers.buffers);
            tailCpu.close();
            tailCpu = null;

            NnApiDelegate.Options nnapiOptions = new NnApiDelegate.Options()
                    .setAcceleratorName("google-edgetpu")
                    .setUseNnapiCpu(false)
                    .setExecutionPreference(NnApiDelegate.Options.EXECUTION_PREFERENCE_SUSTAINED_SPEED);
            nnapi = new NnApiDelegate(nnapiOptions);
            Interpreter.Options tailOptions = new Interpreter.Options()
                    .setNumThreads(4)
                    .setUseXNNPACK(false)
                    .addDelegate(nnapi);
            tail = new Interpreter(tailBuffer, tailOptions);
            validateTailModel(tail, tailMeta, inputQuantization, outputQuantizations, groupToTensor);
            TailBuffers tpuBuffers = allocateTailBuffers(tail);
            ByteBuffer tailInput = allocateInput(tail.getInputTensor(0).numBytes());
            float inputScale = tail.getInputTensor(0).quantizationParams().getScale();
            int inputZero = tail.getInputTensor(0).quantizationParams().getZeroPoint();
            if (inputScale <= 0) throw new IllegalStateException("invalid tail input quantization");

            fillQuantizedTail(input, tailInput, inputScale, inputZero);
            runTail(tail, tailInput, tpuBuffers);
            byte[][] tpuTailBytes = copyOutputBytes(tpuBuffers.buffers);
            boolean exactParity = true;
            for (int i = 0; i < GROUPS; i++) {
                exactParity &= Arrays.equals(cpuTailBytes[i], tpuTailBytes[i]);
            }
            if (!exactParity) throw new IllegalStateException("M31 CPU/TPU raw output parity failed");

            ByteBuffer compactInput = ByteBuffer.allocateDirect(CHANNELS * 4).order(ByteOrder.nativeOrder());
            ByteBuffer compactOutput = ByteBuffer.allocateDirect(CHANNELS * 4).order(ByteOrder.nativeOrder());
            float[] output = new float[ROWS * CHANNELS];
            long initNs = System.nanoTime() - initStart;

            // First inference is reported separately; warmups then leave both interpreters resident.
            RunTiming first = invoke(input, compact, compactInput, compactOutput,
                    tail, tailInput, tpuBuffers, groupToTensor, output);
            for (int i = 0; i < warmups; i++) {
                invoke(input, compact, compactInput, compactOutput,
                        tail, tailInput, tpuBuffers, groupToTensor, output);
            }

            long[] cpuNs = new long[iterations];
            long[] tailNs = new long[iterations];
            long[] totalNs = new long[iterations];
            for (int i = 0; i < iterations; i++) {
                RunTiming timing = invoke(input, compact, compactInput, compactOutput,
                        tail, tailInput, tpuBuffers, groupToTensor, output);
                cpuNs[i] = timing.cpuNs;
                tailNs[i] = timing.tailNs;
                totalNs[i] = timing.totalNs;
            }

            if (nnapi.hasErrors()) {
                throw new IllegalStateException("NNAPI delegate reported error " + nnapi.getNnapiErrno());
            }

            JSONObject accuracy = accuracy(reference, output);
            accuracy.put("pass", accuracy.getDouble("global_nrms") < 0.05
                    && accuracy.getDouble("max_row_nrms") < 0.05
                    && accuracy.getDouble("max_fixed32_group_nrms") < 0.05);
            JSONObject timing = new JSONObject();
            timing.put("initialization_ms", toMs(initNs));
            timing.put("first_inference_ms", toMs(first.totalNs));
            timing.put("warmup_runs", warmups);
            timing.put("timed_iterations", iterations);
            timing.put("compact_cpu", summarize(cpuNs));
            timing.put("tail_quantize_tpu_dequantize", summarize(tailNs));
            timing.put("combined_sequential", summarize(totalNs));

            JSONObject report = new JSONObject();
            report.put("device_model", Build.MODEL);
            report.put("android_release", Build.VERSION.RELEASE);
            report.put("compact_model", COMPACT_MODEL);
            report.put("tail_model", TAIL_MODEL);
            report.put("compact_cpu_delegate", "XNNPACK");
            report.put("tail_delegate", "NNAPI google-edgetpu; NNAPI CPU disabled");
            report.put("tail_cpu_tpu_exact_byte_parity", exactParity);
            report.put("tail_nnapi_errno", nnapi.getNnapiErrno());
            report.put("accuracy", accuracy);
            report.put("timing", timing);
            report.put("measurement_scope", "same-process sequential FFN pair; not end-to-end decoder RTF");
            String rendered = report.toString(2);
            File resultFile = new File(dir, "MIXED-INPROCESS-RESULT.json");
            try (FileOutputStream out = new FileOutputStream(resultFile)) {
                out.write((rendered + "\n").getBytes(StandardCharsets.UTF_8));
                out.getFD().sync();
            }
            Log.i("MixedFfnRuntime", "MIXED_FFN_RESULT " + rendered.replace('\n', ' '));
            if (!accuracy.getBoolean("pass")) {
                throw new IllegalStateException("mixed FFN accuracy screen failed: " + rendered);
            }
            return rendered;
        } finally {
            if (tailCpu != null) tailCpu.close();
            if (tail != null) tail.close();
            if (nnapi != null) nnapi.close();
            if (compact != null) compact.close();
        }
    }

    private static RunTiming invoke(float[] input, Interpreter compact,
            ByteBuffer compactInput, ByteBuffer compactOutput,
            Interpreter tail, ByteBuffer tailInput, TailBuffers tailBuffers,
            int[] groupToTensor, float[] output) {
        long totalStart = System.nanoTime();
        long cpuStart = totalStart;
        compactInput.clear();
        for (int c = 0; c < CHANNELS; c++) compactInput.putFloat(input[c]);
        compactInput.flip();
        compactOutput.clear();
        compact.run(compactInput, compactOutput);
        compactOutput.position(0);
        for (int c = 0; c < CHANNELS; c++) output[c] = compactOutput.getFloat();
        long cpuEnd = System.nanoTime();

        long tailStart = cpuEnd;
        fillQuantizedTail(input, tailInput,
                tail.getInputTensor(0).quantizationParams().getScale(),
                tail.getInputTensor(0).quantizationParams().getZeroPoint());
        runTail(tail, tailInput, tailBuffers);
        dequantizeTail(tail, tailBuffers.buffers, groupToTensor, output);
        long tailEnd = System.nanoTime();
        return new RunTiming(cpuEnd - cpuStart, tailEnd - tailStart, tailEnd - totalStart);
    }

    private static void fillQuantizedTail(float[] input, ByteBuffer buffer, float scale, int zeroPoint) {
        buffer.clear();
        for (int row = 1; row < ROWS; row++) {
            int base = row * CHANNELS;
            for (int c = 0; c < CHANNELS; c++) {
                int q = Math.round(input[base + c] / scale) + zeroPoint;
                q = Math.max(-128, Math.min(127, q));
                buffer.put((byte) q);
            }
        }
        buffer.flip();
    }

    private static void runTail(Interpreter interpreter, ByteBuffer input, TailBuffers outputs) {
        input.position(0);
        input.limit(input.capacity());
        for (ByteBuffer output : outputs.buffers) output.clear();
        interpreter.runForMultipleInputsOutputs(new Object[]{input}, outputs.map);
    }

    private static void dequantizeTail(Interpreter interpreter, ByteBuffer[] buffers,
            int[] groupToTensor, float[] output) {
        for (int group = 0; group < GROUPS; group++) {
            int tensorIndex = groupToTensor[group];
            Tensor.QuantizationParams quant = interpreter.getOutputTensor(tensorIndex).quantizationParams();
            ByteBuffer values = buffers[tensorIndex].duplicate().order(ByteOrder.nativeOrder());
            values.position(0);
            for (int row = 0; row < TAIL_ROWS; row++) {
                int outBase = (row + 1) * CHANNELS + group * GROUP;
                for (int c = 0; c < GROUP; c++) {
                    int q = values.get();
                    output[outBase + c] = (q - quant.getZeroPoint()) * quant.getScale();
                }
            }
        }
    }

    private static ByteBuffer allocateInput(int bytes) {
        return ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder());
    }

    private static TailBuffers allocateTailBuffers(Interpreter interpreter) {
        int count = interpreter.getOutputTensorCount();
        if (count != GROUPS) throw new IllegalStateException("expected 16 tail output tensors, got " + count);
        ByteBuffer[] buffers = new ByteBuffer[count];
        Map<Integer, Object> map = new HashMap<>();
        for (int i = 0; i < count; i++) {
            Tensor tensor = interpreter.getOutputTensor(i);
            requireTensor(tensor, DataType.INT8, new int[]{1, 1, TAIL_ROWS, GROUP}, "tail output " + i);
            buffers[i] = ByteBuffer.allocateDirect(tensor.numBytes()).order(ByteOrder.nativeOrder());
            map.put(i, buffers[i]);
        }
        return new TailBuffers(buffers, map);
    }

    private static void validateTailModel(Interpreter interpreter, JSONObject meta,
            JSONArray manifestInputQuant, JSONArray manifestOutputQuant, int[] groupToTensor) throws Exception {
        Tensor input = interpreter.getInputTensor(0);
        requireTensor(input, DataType.INT8, new int[]{1, 1, TAIL_ROWS, CHANNELS}, "tail input");
        Tensor.QuantizationParams q = input.quantizationParams();
        if (Math.abs(q.getScale() - (float) manifestInputQuant.getDouble(0)) > 1e-7
                || q.getZeroPoint() != manifestInputQuant.getInt(1)) {
            throw new IllegalStateException("tail input quantization differs from manifest");
        }
        if (interpreter.getOutputTensorCount() != GROUPS || groupToTensor.length != GROUPS
                || manifestOutputQuant.length() != GROUPS) {
            throw new IllegalStateException("tail output manifest inventory mismatch");
        }
        for (int i = 0; i < GROUPS; i++) {
            Tensor tensor = interpreter.getOutputTensor(i);
            Tensor.QuantizationParams actual = tensor.quantizationParams();
            JSONArray expected = manifestOutputQuant.getJSONArray(i);
            if (Math.abs(actual.getScale() - (float) expected.getDouble(0)) > 1e-7
                    || actual.getZeroPoint() != expected.getInt(1)) {
                throw new IllegalStateException("tail output quantization differs at tensor " + i);
            }
        }
        boolean[] seen = new boolean[GROUPS];
        for (int index : groupToTensor) {
            if (index < 0 || index >= GROUPS || seen[index]) {
                throw new IllegalStateException("invalid output group mapping");
            }
            seen[index] = true;
        }
    }

    private static void requireTensor(Tensor tensor, DataType type, int[] shape, String label) {
        if (tensor.dataType() != type || !Arrays.equals(tensor.shape(), shape)) {
            throw new IllegalStateException(label + " has " + tensor.dataType() + " "
                    + Arrays.toString(tensor.shape()) + ", expected " + type + " "
                    + Arrays.toString(shape));
        }
    }

    private static JSONObject findTailMeta(File manifest) throws Exception {
        JSONArray all = new JSONArray(new String(Files.readAllBytes(manifest.toPath()), StandardCharsets.UTF_8));
        for (int i = 0; i < all.length(); i++) {
            JSONObject item = all.getJSONObject(i);
            if (TAIL_MODEL.equals(item.optString("variant"))) return item;
        }
        throw new IllegalStateException("tail model entry missing from manifest");
    }

    private static int[] readGroupMapping(JSONObject meta) throws Exception {
        JSONArray array = meta.getJSONArray("output_group_to_tensor_index");
        int[] result = new int[array.length()];
        for (int i = 0; i < result.length; i++) result[i] = array.getInt(i);
        return result;
    }

    private static MappedByteBuffer map(File file) throws Exception {
        try (FileInputStream in = new FileInputStream(file);
             FileChannel channel = in.getChannel()) {
            return channel.map(FileChannel.MapMode.READ_ONLY, 0, file.length());
        }
    }

    private static float[] readFloat32(File file, int count) throws Exception {
        byte[] bytes = Files.readAllBytes(file.toPath());
        if (bytes.length != count * 4) throw new IllegalStateException(file + " has wrong byte count");
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        float[] result = new float[count];
        for (int i = 0; i < count; i++) {
            result[i] = buffer.getFloat();
            if (!Float.isFinite(result[i])) throw new IllegalStateException(file + " contains nonfinite data");
        }
        return result;
    }

    private static byte[][] copyOutputBytes(ByteBuffer[] buffers) {
        byte[][] result = new byte[buffers.length][];
        for (int i = 0; i < buffers.length; i++) {
            ByteBuffer copy = buffers[i].duplicate();
            copy.position(0);
            result[i] = new byte[copy.remaining()];
            copy.get(result[i]);
        }
        return result;
    }

    private static JSONObject accuracy(float[] reference, float[] actual) throws Exception {
        double sumError = 0, sumReference = 0, maxAbs = 0;
        double maxRow = 0, maxGroup = 0, firstRow = 0, tail = 0;
        for (int i = 0; i < reference.length; i++) {
            double delta = actual[i] - reference[i];
            sumError += delta * delta;
            sumReference += (double) reference[i] * reference[i];
            maxAbs = Math.max(maxAbs, Math.abs(delta));
        }
        double global = Math.sqrt(sumError / Math.max(sumReference, 1e-24));
        for (int row = 0; row < ROWS; row++) {
            double e = 0, r = 0;
            for (int c = 0; c < CHANNELS; c++) {
                int i = row * CHANNELS + c;
                double d = actual[i] - reference[i];
                e += d * d;
                r += (double) reference[i] * reference[i];
            }
            double value = Math.sqrt(e / Math.max(r, 1e-24));
            maxRow = Math.max(maxRow, value);
            if (row == 0) firstRow = value;
            else tail = Math.max(tail, value);
        }
        for (int group = 0; group < GROUPS; group++) {
            double e = 0, r = 0;
            for (int row = 0; row < ROWS; row++) {
                for (int c = 0; c < GROUP; c++) {
                    int i = row * CHANNELS + group * GROUP + c;
                    double d = actual[i] - reference[i];
                    e += d * d;
                    r += (double) reference[i] * reference[i];
                }
            }
            maxGroup = Math.max(maxGroup, Math.sqrt(e / Math.max(r, 1e-24)));
        }
        JSONObject result = new JSONObject();
        result.put("global_nrms", global);
        result.put("max_row_nrms", maxRow);
        result.put("first_row_nrms", firstRow);
        result.put("max_tail_row_nrms", tail);
        result.put("max_fixed32_group_nrms", maxGroup);
        result.put("max_abs", maxAbs);
        return result;
    }

    private static JSONObject summarize(long[] samples) throws Exception {
        long[] sorted = samples.clone();
        Arrays.sort(sorted);
        long sum = 0, max = sorted[sorted.length - 1];
        for (long value : samples) sum += value;
        JSONObject result = new JSONObject();
        result.put("average_ms", toMs(sum / (double) samples.length));
        result.put("median_ms", toMs(percentile(sorted, 50)));
        result.put("p95_ms", toMs(percentile(sorted, 95)));
        result.put("p99_ms", toMs(percentile(sorted, 99)));
        result.put("max_ms", toMs(max));
        return result;
    }

    private static long percentile(long[] sorted, int percentile) {
        int index = (int) Math.ceil(percentile / 100.0 * sorted.length) - 1;
        return sorted[Math.max(0, Math.min(sorted.length - 1, index))];
    }

    private static double toMs(double nanos) {
        return nanos / 1_000_000.0;
    }

    private static final class TailBuffers {
        final ByteBuffer[] buffers;
        final Map<Integer, Object> map;
        TailBuffers(ByteBuffer[] buffers, Map<Integer, Object> map) {
            this.buffers = buffers;
            this.map = map;
        }
    }

    private static final class RunTiming {
        final long cpuNs, tailNs, totalNs;
        RunTiming(long cpuNs, long tailNs, long totalNs) {
            this.cpuNs = cpuNs;
            this.tailNs = tailNs;
            this.totalNs = totalNs;
        }
    }
}
