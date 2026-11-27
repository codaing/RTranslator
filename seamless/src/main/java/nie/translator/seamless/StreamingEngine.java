package nie.translator.seamless;

import ai.onnxruntime.*;
import java.io.File;
import java.nio.LongBuffer;
import java.util.*;

/** Experimental full-prefix Seamless streaming inference. Single worker owns sessions. */
final class StreamingEngine implements AutoCloseable {
    interface Sink { void audio(float[] samples) throws Exception; }
    static final int MAX_FRAMES=6000, MIN_FRAMES=192, MAX_TOKENS=100;
    private final OrtEnvironment env=OrtEnvironment.getEnvironment();
    private OrtSession encoder,decoder,units;
    private EnglishUnitVocoder vocoder;
    private final TokenMetadata tokens;
    private final StreamingFbank fbank=new StreamingFbank();
    private final List<float[]> frames=new ArrayList<>();
    private final List<Long> accepted=new ArrayList<>();
    private int durationStart=0, lastFrameCount=0;
    StreamingEngine(File dir)throws Exception{
        tokens=new TokenMetadata(new File(dir,"tokenizer.json"));
        for(long x:tokens.prefix)accepted.add(x);
        try{
            encoder=open(dir,"encoder.int8.onnx");
            decoder=open(dir,"decoder.int8.onnx");
            units=open(dir,"units.onnx");
            vocoder=new EnglishUnitVocoder(new File(dir,"vocoder.onnx").getAbsolutePath());
        }catch(Exception|OutOfMemoryError e){try{close();}catch(Exception ignored){}throw e;}
    }
    private OrtSession open(File dir,String name)throws OrtException{
        try(OrtSession.SessionOptions o=new OrtSession.SessionOptions()){
            o.setIntraOpNumThreads(4);o.setInterOpNumThreads(1);
            return env.createSession(new File(dir,name).getAbsolutePath(),o);
        }
    }
    private long[] prefix(boolean comma){
        long[] t=new long[accepted.size()+(comma?1:0)];
        for(int i=0;i<accepted.size();i++)t[i]=accepted.get(i);
        if(comma)t[t.length-1]=tokens.comma;
        return t;
    }
    private OnnxTensor longs(long[] v)throws OrtException{
        return OnnxTensor.createTensor(env,LongBuffer.wrap(v),new long[]{1,v.length});
    }
    private static Object value(OrtSession.Result r,String name)throws OrtException{
        return r.get(name).orElseThrow(()->new IllegalStateException("missing output "+name)).getValue();
    }
    private static final class Prediction {
        final int token;final float p;final float[][][] features;
        Prediction(int t,float p,float[][][] f){token=t;this.p=p;features=f;}
    }
    private Prediction predict(long[] prefix,float[][][] encoded)throws Exception{
        try(OnnxTensor t=longs(prefix);OnnxTensor e=OnnxTensor.createTensor(env,encoded);
            OrtSession.Result r=decoder.run(Map.of("tokens",t,"encoded",e))){
            float[][][] logits=(float[][][])value(r,"logits");
            return new Prediction(StreamingPolicy.argmax(logits[0][logits[0].length-1]),
                StreamingPolicy.probability((float[][][])value(r,"p_choose")),
                (float[][][])value(r,"features"));
        }
    }
    void push(short[] pcm,boolean finished,Sink sink)throws Exception{
        frames.addAll(fbank.push(pcm));
        if(frames.size()>MAX_FRAMES)throw new IllegalStateException("60-second experimental context limit reached; no audio silently discarded");
        int count=frames.size()&~1;
        if(count<2||(!finished&&count<MIN_FRAMES)||(!finished&&count==lastFrameCount))return;
        lastFrameCount=count;
        float[][][] encoded;
        try(OnnxTensor x=OnnxTensor.createTensor(env,new float[][][]{frames.subList(0,count).toArray(new float[0][])});
            OrtSession.Result r=encoder.run(Collections.singletonMap("fbank",x))){
            encoded=(float[][][])value(r,"encoded");
        }
        int oldSize=accepted.size();
        for(int i=0;i<50;i++){
            Prediction p=predict(prefix(false),encoded);
            if(!StreamingPolicy.write(p.p,p.token,tokens.eos,finished))break;
            if(accepted.size()-tokens.prefix.length>=MAX_TOKENS)
                throw new IllegalStateException("100-token experimental context limit reached");
            accepted.add((long)p.token);
        }
        if(accepted.size()-tokens.prefix.length<2||(!finished&&oldSize==accepted.size()))return;
        long[] t=prefix(true);
        TokenMetadata.Mapping m=tokens.map(t);
        if(m.chars.length==0)return;
        Prediction p=predict(t,encoded);
        if(p.features[0].length!=t.length)throw new IllegalStateException("decoder feature alignment");
        try(OnnxTensor f=OnnxTensor.createTensor(env,p.features);OnnxTensor ch=longs(m.chars);
            OnnxTensor len=longs(m.lengths);OrtSession.Result r=units.run(Map.of("features",f,"chars",ch,"char_lens",len))){
            long[] d=((long[][])value(r,"durations"))[0];
            if(durationStart>d.length)throw new IllegalStateException("duration prefix shrank");
            int total=StreamingPolicy.offset(d,d.length),start=StreamingPolicy.offset(d,durationStart);
            if(total-start<(finished?1:50))return;
            long[] u=StreamingPolicy.units(((float[][][])value(r,"logits"))[0],d,durationStart);
            float[] waveform=vocoder.decode(u);
            if(waveform.length==0)throw new IllegalStateException("empty synthesized output");
            durationStart=Math.max(0,d.length-1);
            sink.audio(waveform);
        }
    }
    @Override public void close()throws Exception{
        Exception first=null;
        for(AutoCloseable c:new AutoCloseable[]{vocoder,units,decoder,encoder})if(c!=null)try{c.close();}catch(Exception e){if(first==null)first=e;}
        if(first!=null)throw first;
    }
}
