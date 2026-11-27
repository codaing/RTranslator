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
    @Test public void fourModelStreamingChainProducesAudioBeforeEof() throws Exception {
        // Synthetic graphs validate application wiring, not translation quality.
        android.content.Context context=InstrumentationRegistry.getInstrumentation().getContext();
        File parent=new File(InstrumentationRegistry.getInstrumentation().getTargetContext().getCacheDir(),"stream-test");
        ModelPack.clear(parent);assertTrue(parent.mkdir());
        File dir=new File(parent,"models");
        try {
            try(InputStream in=context.getAssets().open("test-model-pack.zip")){ModelPack.importZip(in,dir);}
            assertTrue(ModelPack.present(dir));
            assertTrue(new File(dir,"encoder.int8.onnx.weights").isFile());
            java.util.List<float[]> output=new java.util.ArrayList<>();
            try(StreamingEngine engine=new StreamingEngine(dir,false)){
                short[] pcm=new short[5120];
                for(int i=0;i<6;i++)engine.push(pcm,false,output::add);
                assertTrue("must wait for minimum context",output.isEmpty());
                engine.push(pcm,false,output::add);
                assertEquals(2,engine.targetTokens());
                assertEquals("audio must be produced before EOF",1,output.size());
                assertEquals(60,output.get(0).length);
                for(int i=0;i<60;i++)assertEquals(i<30?0.01f:0.02f,output.get(0)[i],0.00001f);
                engine.push(pcm,false,output::add);
                engine.push(new short[0],true,output::add);
                assertEquals("no duplicate output after EOF",1,output.size());
            }
        } finally {ModelPack.clear(parent);}
    }

    @Test public void corruptedReplacementDoesNotDestroyInstalledModels() throws Exception {
        android.content.Context context=InstrumentationRegistry.getInstrumentation().getContext();
        File parent=new File(InstrumentationRegistry.getInstrumentation().getTargetContext().getCacheDir(),"import-test");
        ModelPack.clear(parent);assertTrue(parent.mkdir());
        File dir=new File(parent,"models");
        try {
            try(InputStream in=context.getAssets().open("test-model-pack.zip")){ModelPack.importZip(in,dir);}
            try(InputStream in=context.getAssets().open("test-corrupt-pack.zip")){
                ModelPack.importZip(in,dir);
                fail("corrupted pack accepted");
            }catch(java.io.IOException expected){assertTrue(expected.getMessage().contains("SHA256 mismatch"));}
            assertTrue(ModelPack.present(dir));
            ModelPack.verify(dir);
        }finally{ModelPack.clear(parent);}
    }
}
