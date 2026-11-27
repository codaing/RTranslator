package nie.translator.seamless;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import org.junit.Test;
import org.junit.runner.RunWith;
import static org.junit.Assert.*;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;

@RunWith(AndroidJUnit4.class)
public class RuntimeTest {
    @Test public void vocoderInterfaceExecutesOnAndroid() throws Exception {
        // A small deterministic graph tests the real Java/ORT interface, not speech quality.
        android.content.Context context = InstrumentationRegistry.getInstrumentation().getContext();
        File f = new File(InstrumentationRegistry.getInstrumentation().getTargetContext().getCacheDir(), "test-vocoder.onnx");
        try (InputStream in = context.getAssets().open("test-vocoder.onnx");
             FileOutputStream out = new FileOutputStream(f)) {
            byte[] bytes = new byte[4096];
            int count;
            while ((count = in.read(bytes)) != -1) out.write(bytes, 0, count);
        }
        try (EnglishUnitVocoder model = new EnglishUnitVocoder(f.getAbsolutePath())) {
            assertArrayEquals(new float[]{0.01f, 0.02f, 0.03f}, model.decode(new long[]{1,2,3}), 0.00001f);
            assertEquals(0, model.decode(new long[0]).length);
            try { model.decode(new long[]{-1}); fail("Invalid units accepted"); }
            catch (IllegalArgumentException expected) {}
        } finally { f.delete(); }
    }
}
