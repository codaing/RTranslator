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
    private int durationStart=0, lastFrameCount=0, rollovers=0;
    private final boolean adaptive;
    private final ContextBoundary boundary=new ContextBoundary();
    int contextFrames(){return frames.size();}
    int targetTokens(){return accepted.size()-tokens.prefix.length;}
    int rollovers(){return rollovers;}
    StreamingEngine(File dir,boolean adaptive)throws Exception{
        this.adaptive=adaptive;
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
        boundary.observe(pcm);
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
        boolean eos=false;float eosProbability=0;
        for(int i=0;i<50;i++){
            Prediction p=predict(prefix(false),encoded);
            if(p.token==tokens.eos){eos=true;eosProbability=p.p;break;}
            if(!StreamingPolicy.write(p.p,p.token,tokens.eos,finished))break;
            if(accepted.size()-tokens.prefix.length>=MAX_TOKENS)
                throw new IllegalStateException("100-token experimental context limit reached");
            accepted.add((long)p.token);
        }
        boolean rollover=!finished&&boundary.candidate(adaptive,count,targetTokens(),eos,eosProbability);
        if(targetTokens()<2||(!finished&&!rollover&&oldSize==accepted.size()))return;
        boolean flushed=synthesize(encoded,finished||rollover,sink);
        // This is an experimental EOS + quiet boundary heuristic, not proof
        // of source coverage. Reset only after successful output handoff.
        if(rollover&&flushed){
            frames.subList(0,count).clear(); // keep any unencoded odd frame
            accepted.clear();for(long x:tokens.prefix)accepted.add(x);
            durationStart=0;lastFrameCount=0;boundary.reset();rollovers++;
        }
    }
    private boolean synthesize(float[][][] encoded,boolean flush,Sink sink)throws Exception{
        long[] t=prefix(true);
        TokenMetadata.Mapping m=tokens.map(t);
        if(m.chars.length==0)return false;
        Prediction p=predict(t,encoded);
        if(p.features[0].length!=t.length)throw new IllegalStateException("decoder feature alignment");
        try(OnnxTensor f=OnnxTensor.createTensor(env,p.features);OnnxTensor ch=longs(m.chars);
            OnnxTensor len=longs(m.lengths);OrtSession.Result r=units.run(Map.of("features",f,"chars",ch,"char_lens",len))){
            long[] d=((long[][])value(r,"durations"))[0];
            if(durationStart>d.length)throw new IllegalStateException("duration prefix shrank");
            int total=StreamingPolicy.offset(d,d.length),start=StreamingPolicy.offset(d,durationStart);
            if(total-start==0)return flush;
            if(total-start<(flush?1:50))return false;
            long[] u=StreamingPolicy.units(((float[][][])value(r,"logits"))[0],d,durationStart);
            float[] waveform=vocoder.decode(u);
            if(waveform.length==0)throw new IllegalStateException("empty synthesized output");
            sink.audio(waveform);
            durationStart=flush?d.length:Math.max(0,d.length-1);
            return true;
        }
    }
    @Override public void close()throws Exception{
        Exception first=null;
        for(AutoCloseable c:new AutoCloseable[]{vocoder,units,decoder,encoder})if(c!=null)try{c.close();}catch(Exception e){if(first==null)first=e;}
        if(first!=null)throw first;
    }
}
