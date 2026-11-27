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
    @Test public void xnnpackMatmulProviderExecutesOnAndroid() throws Exception {
        android.content.Context context=InstrumentationRegistry.getInstrumentation().getContext();
        byte[] model;
        try(InputStream in=context.getAssets().open("test-xnnpack-matmul.onnx");
            java.io.ByteArrayOutputStream out=new java.io.ByteArrayOutputStream()){
            byte[] b=new byte[4096];int n;while((n=in.read(b))!=-1)out.write(b,0,n);model=out.toByteArray();
        }
        ai.onnxruntime.OrtEnvironment env=ai.onnxruntime.OrtEnvironment.getEnvironment();
        try(ai.onnxruntime.OrtSession.SessionOptions opts=new ai.onnxruntime.OrtSession.SessionOptions()){
            opts.setIntraOpNumThreads(1);
            opts.addConfigEntry("session.intra_op.allow_spinning","0");
            opts.addXnnpack(java.util.Collections.singletonMap("intra_op_num_threads","2"));
            try(ai.onnxruntime.OrtSession session=env.createSession(model,opts);
                ai.onnxruntime.OnnxTensor x=ai.onnxruntime.OnnxTensor.createTensor(env,new float[][]{{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}});
                ai.onnxruntime.OrtSession.Result result=session.run(java.util.Collections.singletonMap("x",x))){
                float[][] y=(float[][])result.get(0).getValue();
                for(int i=0;i<16;i++)assertEquals((i+1)*2f,y[0][i],0.0001f);
            }
        }
    }
    @Test public void fourModelStreamingChainProducesAudioBeforeEof() throws Exception {
        chain("test-model-pack.zip",false,false);
    }
    @Test public void cachedDecoderProducesAudioBeforeEof() throws Exception {
        chain("test-incremental-pack.zip",true,false);
    }
    @Test public void quietCpuCachedDecoderPreservesSyntheticAudio() throws Exception {
        chain("test-incremental-pack.zip",true,true);
    }
    private void chain(String pack,boolean cached,boolean quiet)throws Exception{
        // Synthetic graphs validate application wiring, not translation quality.
        android.content.Context context=InstrumentationRegistry.getInstrumentation().getContext();
        File parent=new File(InstrumentationRegistry.getInstrumentation().getTargetContext().getCacheDir(),"stream-test");
        ModelPack.clear(parent);assertTrue(parent.mkdir());
        File dir=new File(parent,"models");
        try {
            try(InputStream in=context.getAssets().open(pack)){ModelPack.importZip(in,dir);}
            assertTrue(ModelPack.present(dir));
            assertEquals(cached,ModelPack.incremental(dir));
            assertTrue(new File(dir,"encoder.int8.onnx.weights").isFile());
            java.util.List<float[]> output=new java.util.ArrayList<>();
            try(StreamingEngine engine=new StreamingEngine(dir,false,false,false,quiet)){
                assertEquals(cached?"incremental-fp16-kv-cache":"full-prefix-int8",engine.decoderMode());
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
                assertTrue(engine.stats.calls(StageStats.ENCODER)>0);
                assertTrue(engine.stats.calls(StageStats.PREFILL)>0);
                assertTrue(engine.stats.calls(StageStats.COMMA)>0);
                assertTrue(engine.stats.calls(StageStats.UNITS)>0);
                assertEquals(1,engine.stats.calls(StageStats.VOCODER));
                if(cached)assertTrue(engine.stats.calls(StageStats.STEP)>0);
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
    @Test public void probeNativeLowBitMatmul() throws Exception {
        android.content.Context context=InstrumentationRegistry.getInstrumentation().getContext();
        ai.onnxruntime.OrtEnvironment env=ai.onnxruntime.OrtEnvironment.getEnvironment();
        int supported=0;
        for(int bits:new int[]{4,8}){
            String file="test-matmul-"+bits+"bit.onnx";
            byte[] model;
            try(InputStream in=context.getAssets().open(file);
                java.io.ByteArrayOutputStream out=new java.io.ByteArrayOutputStream()){
                byte[] b=new byte[4096];int n;
                while((n=in.read(b))!=-1)out.write(b,0,n);
                model=out.toByteArray();
            }
            try(ai.onnxruntime.OrtSession.SessionOptions opts=new ai.onnxruntime.OrtSession.SessionOptions()){
                opts.setIntraOpNumThreads(2);
                ai.onnxruntime.OrtSession session;
                try{session=env.createSession(model,opts);}
                catch(ai.onnxruntime.OrtException unavailable){
                    android.util.Log.w("LowBitProbe","bits="+bits+" session unavailable: "+unavailable.getMessage());
                    continue;
                }
                try(ai.onnxruntime.OrtSession s=session){
                    float[][] input=new float[2][32];
                    java.util.Arrays.fill(input[0],1f);
                    java.util.Arrays.fill(input[1],-0.5f);
                    try(ai.onnxruntime.OnnxTensor x=ai.onnxruntime.OnnxTensor.createTensor(env,input);
                        ai.onnxruntime.OrtSession.Result result=s.run(java.util.Collections.singletonMap("x",x))){
                        float[][] y=(float[][])result.get(0).getValue();
                        assertArrayEquals(new float[]{8f,-16f},y[0],0.001f);
                        assertArrayEquals(new float[]{-4f,8f},y[1],0.001f);
                    }
                    supported++;
                    android.util.Log.i("LowBitProbe","bits="+bits+" numeric probe PASS");
                }
            }
        }
        assertTrue("Neither packed low-bit operator could load",supported>0);
    }
}
