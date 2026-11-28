package nie.translator.seamless;

import ai.onnxruntime.*;
import java.util.Collections;
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
    ALL(15,"WebGPU all stages");
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
        if(this!=CPU&&!OrtEnvironment.getAvailableProviders().toString().contains("WEBGPU"))
            throw new IllegalStateException("WebGPU is not available in this Android runtime");
    }
    void configure(OrtSession.SessionOptions options,String name)throws OrtException{
        if(!uses(name))return;
        options.addWebGPU(Collections.emptyMap());
        android.util.Log.i("SeamlessModels","WebGPU requested for "+name+"; CPU fallback allowed");
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
