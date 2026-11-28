package nie.translator.seamless;

import android.os.Build;
import android.util.Log;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;

import org.json.JSONObject;
import org.junit.Assume;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.io.File;

import static org.junit.Assert.assertTrue;

/** Pixel-only in-process CPU + Edge TPU mixed FFN test; skipped on generic CI emulators. */
@RunWith(AndroidJUnit4.class)
public class MixedFfnRuntimeTest {
    @Test
    public void compactCpuThenEdgeTpuTailInSameProcess() throws Exception {
        Assume.assumeTrue("requires Pixel 9a google-edgetpu NNAPI device",
                Build.MODEL != null && Build.MODEL.toLowerCase(java.util.Locale.US).contains("pixel 9a"));
        android.content.Context target = InstrumentationRegistry.getInstrumentation().getTargetContext();
        File directory = new File(target.getFilesDir(), "seamless-mixed");
        String rendered = MixedFfnBenchmark.run(directory, 50, 1000);
        JSONObject report = new JSONObject(rendered);
        assertTrue("CPU/TPU byte parity failed", report.getBoolean("tail_cpu_tpu_exact_byte_parity"));
        assertTrue("accuracy screen failed", report.getJSONObject("accuracy").getBoolean("pass"));
        assertTrue("NNAPI reported an error", report.getInt("tail_nnapi_errno") == 0);
        Log.i("MixedFfnRuntime", "MIXED_FFN_INPROCESS_PASS");
    }
}
