package nie.translator.seamless;

import ai.onnxruntime.*;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import org.junit.Test;
import org.junit.runner.RunWith;
import static org.junit.Assert.*;
import java.io.*;
import java.util.*;
import java.lang.reflect.InvocationTargetException;

@RunWith(AndroidJUnit4.class)
public class WebGpuCapabilityProbe {
    @Test public void probeBundledProviderWithoutCpuFallback() throws Exception {
        OrtEnvironment env=OrtEnvironment.getEnvironment();
        String providers=OrtEnvironment.getAvailableProviders().toString();
        android.util.Log.i("WebGpuProbe","availableProviders="+providers);
        byte[] model;
        try(InputStream in=InstrumentationRegistry.getInstrumentation().getContext().getAssets().open("test-xnnpack-matmul.onnx");
            ByteArrayOutputStream out=new ByteArrayOutputStream()){
            byte[] b=new byte[4096];int n;while((n=in.read(b))!=-1)out.write(b,0,n);model=out.toByteArray();
        }
        java.lang.reflect.Method method;
        try {
            method=OrtSession.SessionOptions.class.getMethod("addWebGPU",Map.class);
        } catch(NoSuchMethodException e){
            android.util.Log.i("WebGpuProbe","RESULT=JAVA_API_UNAVAILABLE");return;
        }
        if(!providers.toLowerCase(Locale.ROOT).contains("webgpu")){
            android.util.Log.i("WebGpuProbe","RESULT=PROVIDER_NOT_ADVERTISED");return;
        }
        try(OrtSession.SessionOptions opts=new OrtSession.SessionOptions()){
            opts.setIntraOpNumThreads(1);
            opts.addConfigEntry("session.disable_cpu_ep_fallback","1");
            try {
                method.invoke(opts,Collections.emptyMap());
            } catch(InvocationTargetException e){
                android.util.Log.i("WebGpuProbe","RESULT=REGISTRATION_FAILED "+e.getCause());return;
            }
            try(OrtSession session=env.createSession(model,opts);
                OnnxTensor x=OnnxTensor.createTensor(env,new float[][]{{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}});
                OrtSession.Result result=session.run(Collections.singletonMap("x",x))){
                float[][] y=(float[][])result.get(0).getValue();
                for(int i=0;i<16;i++)assertEquals((i+1)*2f,y[0][i],0.0001f);
                android.util.Log.i("WebGpuProbe","RESULT=MATMUL_PASS_CPU_FALLBACK_DISABLED");
            }catch(OrtException e){
                android.util.Log.i("WebGpuProbe","RESULT=SESSION_OR_EXECUTION_FAILED "+e.getMessage());
            }
        }
    }
}
