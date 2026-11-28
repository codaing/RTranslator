package nie.translator.seamless;

import android.content.Context;
import ai.onnxruntime.*;
import java.io.*;
import java.nio.*;
import java.util.*;

/** Independent Vulkan kernel probe plus an actual-model projection parity test. */
final class VulkanProbe {
    private static byte[] readAsset(Context context,String name)throws IOException{
        try(InputStream in=context.getAssets().open(name);ByteArrayOutputStream out=new ByteArrayOutputStream()){
            byte[] buffer=new byte[65536];int n;
            while((n=in.read(buffer))!=-1){
                if(out.size()+n>128*1024*1024)throw new IOException("asset too large: "+name);
                out.write(buffer,0,n);
            }
            return out.toByteArray();
        }
    }
    static String run(){
        try{
            System.loadLibrary("seamless_vulkan_probe");
            return nativeRun(0,null,null);
        }catch(LinkageError e){return "Direct Vulkan probe unavailable: "+e;}
    }
    static String matmul(int rows){
        try{
            System.loadLibrary("seamless_vulkan_probe");
            return nativeRun(rows,null,null);
        }catch(LinkageError e){return "Vulkan MatMul unavailable: "+e;}
    }
    static String gemv(int dispatches){return matmul(-dispatches);}
    static String subgroup(){return matmul(-201);}
    static String fused(int outputs){return matmul(-(100+outputs));}

    static String realProjection(Context context){
        try{
            System.loadLibrary("seamless_vulkan_probe");
            byte[] fixture=readAsset(context,"vulkan-real-projection.bin");
            byte[] model=readAsset(context,"projection-reference.onnx");
            String metadata=new String(readAsset(context,"vulkan-real-projection.json"),"UTF-8");
            ByteBuffer header=ByteBuffer.wrap(fixture).order(ByteOrder.LITTLE_ENDIAN);
            if(fixture.length<20||header.getInt()!=0x31505653||header.getInt()!=1)
                throw new IOException("invalid real-projection fixture header");
            int m=header.getInt(),k=header.getInt(),n=header.getInt();
            long expectedBytes=20L+(long)m*k*4L+(long)n*k*2L;
            if(m<1||k<2||(k&1)!=0||n<4||n%4!=0||expectedBytes!=fixture.length)
                throw new IOException("invalid real-projection dimensions or file length");
            float[] activation=new float[Math.multiplyExact(m,k)];
            ByteBuffer activationBytes=ByteBuffer.wrap(fixture,20,activation.length*4)
                    .slice().order(ByteOrder.LITTLE_ENDIAN);
            activationBytes.asFloatBuffer().get(activation);

            File modelFile=new File(context.getCacheDir(),"vulkan-projection-reference.onnx");
            try(FileOutputStream out=new FileOutputStream(modelFile)){out.write(model);}
            OrtEnvironment environment=OrtEnvironment.getEnvironment();
            long loadStart=android.os.SystemClock.elapsedRealtimeNanos();
            OrtSession session;
            try(OrtSession.SessionOptions options=new OrtSession.SessionOptions()){
                options.setIntraOpNumThreads(4);options.setInterOpNumThreads(1);
                session=environment.createSession(modelFile.getAbsolutePath(),options);
            }
            double loadMs=(android.os.SystemClock.elapsedRealtimeNanos()-loadStart)/1_000_000.0;
            float[] expected=new float[Math.multiplyExact(m,n)];
            double[] cpuMs=new double[11];
            try(OrtSession cpu=session;
                OnnxTensor input=OnnxTensor.createTensor(environment,FloatBuffer.wrap(activation),new long[]{m,k})){
                for(int i=0;i<14;i++){
                    long start=android.os.SystemClock.elapsedRealtimeNanos();
                    try(OrtSession.Result result=cpu.run(Collections.singletonMap("x",input))){
                        float[][] values=(float[][])result.get(0).getValue();
                        if(values.length!=m||values[0].length!=n)throw new IOException("CPU ORT output shape mismatch");
                        int at=0;
                        for(float[] row:values){
                            if(row.length!=n)throw new IOException("ragged CPU ORT output");
                            for(float value:row){
                                if(!Float.isFinite(value))throw new IOException("non-finite CPU ORT output");
                                expected[at++]=value;
                            }
                        }
                    }
                    double elapsed=(android.os.SystemClock.elapsedRealtimeNanos()-start)/1_000_000.0;
                    if(i>=3)cpuMs[i-3]=elapsed;
                }
            }
            Arrays.sort(cpuMs);
            String gpu=nativeRun(-300,fixture,expected);
            return "Real model projection parity test (actual decoder weights and Mandarin activation)\n"+
                "Fixture: M="+m+" K="+k+" N="+n+"; CPU reference model load "+String.format(Locale.US,"%.2f ms",loadMs)+
                "\n"+metadata+"\nCPU ONNX Runtime CPUExecutionProvider: median "+
                String.format(Locale.US,"%.3f ms",cpuMs[cpuMs.length/2])+"; range "+
                String.format(Locale.US,"%.3f–%.3f ms",cpuMs[0],cpuMs[cpuMs.length-1])+
                " (3 warmups, 11 measured runs)\n"+gpu;
        }catch(Exception|OutOfMemoryError e){return "Real model Vulkan projection: FAIL "+e;}
    }
    private static native String nativeRun(int mode,byte[] fixture,float[] expected);
}
