package nie.translator.seamless;

import ai.onnxruntime.*;
import java.util.Collections;
import java.util.EnumSet;
import ai.onnxruntime.providers.NNAPIFlags;
import java.security.MessageDigest;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/** Opt-in offline diagnostics only. CPU fallback is permitted for mixed real graphs. */
enum GpuTuning {
    CPU(0,"CPU only (reference)"),
    DECODER(2,"WebGPU decoder + source projections"),
    ENCODER(1,"WebGPU encoder"),
    ACOUSTIC(4,"WebGPU UnitY2"),
    FP16(14,"WebGPU decoder + projections + UnitY2 + vocoder"),
    ALL(15,"WebGPU all stages"),
    NNAPI_ENCODER(1,"NNAPI encoder (non-CPU devices)"),
    NNAPI_DECODER(2,"NNAPI decoder + projections (non-CPU devices)"),
    NNAPI_SYNTHESIS(12,"NNAPI acoustic units + vocoder (non-CPU devices)"),
    NNAPI_ALL(15,"NNAPI all stages (non-CPU devices)");
    private final int mask;
    private final String label;
    GpuTuning(int mask,String label){this.mask=mask;this.label=label;}
    @Override public String toString(){return label;}
    boolean uses(String name){
        int bit;
        switch(name){
            case "encoder.int8.onnx":bit=1;break;
            case "decoder.int8.onnx":
            case DecoderCache.GRAPH:
            case DecoderCache.STEP:
            case DecoderCache.SOURCE:bit=2;break;
            case "units.onnx":bit=4;break;
            case "vocoder.onnx":bit=8;break;
            default:throw new IllegalArgumentException("Unknown model: "+name);
        }
        return (mask&bit)!=0;
    }
    void requireAvailable(){
        if(isNnapi()){
            if(!OrtEnvironment.getAvailableProviders().toString().contains("NNAPI"))
                throw new IllegalStateException("NNAPI is not available in this Android runtime");
            return;
        }
        if(this!=CPU&&!OrtEnvironment.getAvailableProviders().toString().contains("WEBGPU"))
            throw new IllegalStateException("WebGPU is not available in this Android runtime");
    }
    void configure(OrtSession.SessionOptions options,String name)throws OrtException{
        if(!uses(name))return;
        if(isNnapi()){
            options.addNnapi(EnumSet.of(NNAPIFlags.CPU_DISABLED));
            android.util.Log.i("SeamlessModels","NNAPI non-CPU devices requested for "+name+"; ORT CPU fallback allowed");
            return;
        }
        options.addWebGPU(Collections.emptyMap());
        android.util.Log.i("SeamlessModels","WebGPU requested for "+name+"; CPU fallback allowed");
    }
    boolean isNnapi(){return name().startsWith("NNAPI_");}
    boolean needsFp16Probe(){return !isNnapi()&&(mask&14)!=0;}
    static boolean missingShaderF16(OrtException e){
        String message=e.getMessage();
        return message!=null&&message.contains("requires f16")&&message.contains("device does not support it");
    }
    /** Execute a real GPU-only graph. A provider listing is not proof of execution. */
    static void probe(android.content.Context context,boolean half)throws Exception{
        probe(context,half,Collections.emptyMap(),false);
    }
    static void probe(android.content.Context context,boolean half,java.util.Map<String,String> providerOptions,
                      boolean optimized)throws Exception{
        if(!OrtEnvironment.getAvailableProviders().toString().contains("WEBGPU"))
            throw new IllegalStateException("WebGPU provider unavailable");
        byte[] model;
        String name=half?"webgpu-fp16-probe.onnx":"webgpu-fp32-probe.onnx";
        try(java.io.InputStream in=context.getAssets().open(name);
            java.io.ByteArrayOutputStream out=new java.io.ByteArrayOutputStream()){
            byte[] b=new byte[4096];int n;
            while((n=in.read(b))!=-1)out.write(b,0,n);
            model=out.toByteArray();
        }
        OrtEnvironment env=OrtEnvironment.getEnvironment();
        try(OrtSession.SessionOptions opts=new OrtSession.SessionOptions()){
            opts.setIntraOpNumThreads(1);
            if(!optimized)opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.NO_OPT);
            opts.addConfigEntry("session.disable_cpu_ep_fallback","1");
            opts.addWebGPU(providerOptions);
            try(OrtSession session=env.createSession(model,opts);
                OnnxTensor x=OnnxTensor.createTensor(env,new float[][]{{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}});
                OrtSession.Result r=session.run(Collections.singletonMap("x",x))){
                float[][] y=(float[][])r.get(0).getValue();
                if(y.length!=1||y[0].length!=16)throw new IllegalStateException("GPU probe shape mismatch");
                for(int i=0;i<16;i++)if(!Float.isFinite(y[0][i])||Math.abs(y[0][i]-(i+1)*2f)>0.0001f)
                    throw new IllegalStateException("GPU probe numeric mismatch at "+i);
            }
        }
    }
    static String fingerprint(long[] ids){
        try{
            MessageDigest digest=MessageDigest.getInstance("SHA-256");
            ByteBuffer bytes=ByteBuffer.allocate(ids.length*8).order(ByteOrder.LITTLE_ENDIAN);
            for(long id:ids)bytes.putLong(id);
            byte[] hash=digest.digest(bytes.array());
            StringBuilder result=new StringBuilder(64);
            for(byte b:hash)result.append(String.format(java.util.Locale.ROOT,"%02x",b&255));
            return result.toString();
        }catch(java.security.NoSuchAlgorithmException e){throw new AssertionError(e);}
    }
}
