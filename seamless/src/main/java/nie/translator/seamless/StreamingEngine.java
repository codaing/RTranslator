package nie.translator.seamless;

import ai.onnxruntime.*;
import java.io.File;
import java.nio.LongBuffer;
import java.util.*;

/** Experimental full-prefix Seamless streaming inference. Single worker owns sessions. */
final class StreamingEngine implements AutoCloseable {
    interface Sink { void audio(float[] samples) throws Exception; }
    static final int MAX_FRAMES=6000, MIN_FRAMES=192, UPDATE_INTERVAL_FRAMES=64, MAX_TOKENS=100;
    private final OrtEnvironment env=OrtEnvironment.getEnvironment();
    private OrtSession encoder,decoder,units;
    private EnglishUnitVocoder vocoder;
    private final TokenMetadata tokens;
    private DecoderCache.Layout cacheLayout;
    String decoderMode(){return cacheLayout==null?"full-prefix-int8":"incremental-fp16-kv-cache";}
    private final StreamingFbank fbank=new StreamingFbank();
    private final List<float[]> frames=new ArrayList<>();
    private final List<Long> accepted=new ArrayList<>();
    private int durationStart=0, lastFrameCount=0, rollovers=0;
    private final boolean adaptive;
    private final boolean xnnpackFp32Stages;
    private final boolean nnapiEncoder;
    private final boolean quietCpu;
    final StageStats stats=new StageStats();
    private final ContextBoundary boundary=new ContextBoundary();
    int contextFrames(){return frames.size();}
    int targetTokens(){return accepted.size()-tokens.prefix.length;}
    int rollovers(){return rollovers;}
    StreamingEngine(File dir,boolean adaptive)throws Exception{
        this(dir,adaptive,false,false,false);
    }
    StreamingEngine(File dir,boolean adaptive,boolean xnnpackFp32Stages,boolean nnapiEncoder)throws Exception{
        this(dir,adaptive,xnnpackFp32Stages,nnapiEncoder,false);
    }
    StreamingEngine(File dir,boolean adaptive,boolean xnnpackFp32Stages,boolean nnapiEncoder,boolean quietCpu)throws Exception{
        this.adaptive=adaptive;this.quietCpu=quietCpu;
        this.xnnpackFp32Stages=xnnpackFp32Stages;
        this.nnapiEncoder=nnapiEncoder;
        tokens=new TokenMetadata(new File(dir,"tokenizer.json"));
        for(long x:tokens.prefix)accepted.add(x);
        try{
            encoder=open(dir,"encoder.int8.onnx");
            if(ModelPack.incremental(dir)){
                cacheLayout=new DecoderCache.Layout(dir);
                decoder=open(dir,DecoderCache.GRAPH);
                cacheLayout.check(decoder);
            }else decoder=open(dir,"decoder.int8.onnx");
            units=open(dir,"units.onnx");
            vocoder=new EnglishUnitVocoder(new File(dir,"vocoder.onnx").getAbsolutePath(),quietCpu);
        }catch(Exception|OutOfMemoryError e){try{close();}catch(Exception ignored){}throw e;}
    }
    private OrtSession open(File dir,String name)throws OrtException{
        try(OrtSession.SessionOptions o=new OrtSession.SessionOptions()){
            o.setIntraOpNumThreads(4);o.setInterOpNumThreads(1);
            if(name.equals("decoder.int8.onnx")){
                // Match the validated native INT8 host session; avoid a second
                // packed-weight copy and allocator high-water retention.
                o.setCPUArenaAllocator(false);
                o.setMemoryPatternOptimization(false);
                o.addConfigEntry("session.disable_prepacking","1");
            }
            if(name.equals("encoder.int8.onnx")&&nnapiEncoder){
                // Experimental: Android NNAPI may claim supported encoder nodes;
                // unsupported operators remain on the default CPU provider.
                o.addNnapi();
                android.util.Log.i("SeamlessModels","NNAPI requested for encoder");
            }
            boolean xnn=xnnpackFp32Stages&&name.equals("units.onnx");
            if(quietCpu)CpuTuning.quiet(o,xnn);
            if(xnn){
                // XNNPACK owns its thread pool; avoid contention with ORT's pool.
                o.setIntraOpNumThreads(1);
                if(!quietCpu)o.addConfigEntry("session.intra_op.allow_spinning","0");
                o.addXnnpack(Collections.singletonMap("intra_op_num_threads","4"));
                android.util.Log.i("SeamlessModels","XNNPACK requested for "+name);
            }
            long start=android.os.SystemClock.elapsedRealtime();
            OrtSession session=env.createSession(new File(dir,name).getAbsolutePath(),o);
            android.util.Log.i("SeamlessModels","loaded "+name+" in "+(android.os.SystemClock.elapsedRealtime()-start)+" ms, PSS="+android.os.Debug.getPss()+" KiB");
            return session;
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
    static final class Prediction {
        final int token;final float p;final float[][][] features;
        Prediction(int t,float p,float[][][] f){token=t;this.p=p;features=f;}
    }
    private Prediction predict(long[] prefix,float[][][] encoded,int phase)throws Exception{
        long started=System.nanoTime();
        try(OnnxTensor t=longs(prefix);OnnxTensor e=OnnxTensor.createTensor(env,encoded);
            OrtSession.Result r=decoder.run(Map.of("tokens",t,"encoded",e))){
            float[][][] logits=(float[][][])value(r,"logits");
            return new Prediction(StreamingPolicy.argmax(logits[0][logits[0].length-1]),
                StreamingPolicy.probability((float[][][])value(r,"p_choose")),
                (float[][][])value(r,"features"));
        }finally{stats.add(phase,started,prefix.length);}
    }
    private Prediction cachedStep(DecoderCache cache,long[] ids,int phase)throws Exception{
        long started=System.nanoTime();
        try{return cache.step(ids);}finally{stats.add(phase,started,ids.length);}
    }
    void push(short[] pcm,boolean finished,Sink sink)throws Exception{
        long fbankStart=System.nanoTime();
        boundary.observe(pcm);
        frames.addAll(fbank.push(pcm));
        stats.add(StageStats.FBANK,fbankStart,pcm.length);
        if(frames.size()>MAX_FRAMES)throw new IllegalStateException("60-second experimental context limit reached; no audio silently discarded");
        int count=frames.size()&~1;
        if(count<2||(!finished&&count<MIN_FRAMES)||
                (!finished&&count-lastFrameCount<UPDATE_INTERVAL_FRAMES))return;
        lastFrameCount=count;
        float[][][] encoded;
        long encoderStart=System.nanoTime();
        try(OnnxTensor x=OnnxTensor.createTensor(env,new float[][][]{frames.subList(0,count).toArray(new float[0][])});
            OrtSession.Result r=encoder.run(Collections.singletonMap("fbank",x))){
            encoded=(float[][][])value(r,"encoded");
        }finally{stats.add(StageStats.ENCODER,encoderStart,count);}
        int oldSize=accepted.size();
        boolean eos=false;float eosProbability=0;
        // Cache is valid only for this encoder output. Native result tensors
        // are held until the next ORT call finishes, then closed.
        try(DecoderCache cache=cacheLayout==null?null:new DecoderCache(env,decoder,cacheLayout,encoded)){
        Prediction p=cache==null?null:cachedStep(cache,prefix(false),StageStats.PREFILL);
        for(int i=0;i<50;i++){
            if(cache==null)p=predict(prefix(false),encoded,StageStats.PREFILL);
            if(p.token==tokens.eos){eos=true;eosProbability=p.p;break;}
            if(!StreamingPolicy.write(p.p,p.token,tokens.eos,finished))break;
            if(accepted.size()-tokens.prefix.length>=MAX_TOKENS)
                throw new IllegalStateException("100-token experimental context limit reached");
            accepted.add((long)p.token);
            if(cache!=null)p=cachedStep(cache,new long[]{p.token},StageStats.STEP);
        }
        boolean rollover=!finished&&boundary.candidate(adaptive,count,targetTokens(),eos,eosProbability);
        if(targetTokens()<2||(!finished&&!rollover&&oldSize==accepted.size()))return;
        boolean flushed=synthesize(encoded,cache,finished||rollover,sink);
        // This is an experimental EOS + quiet boundary heuristic, not proof
        // of source coverage. Reset only after successful output handoff.
        if(rollover&&flushed){
            frames.subList(0,count).clear(); // keep any unencoded odd frame
            accepted.clear();for(long x:tokens.prefix)accepted.add(x);
            durationStart=0;lastFrameCount=0;boundary.reset();rollovers++;
        }
        }
    }
    private boolean synthesize(float[][][] encoded,DecoderCache cache,boolean flush,Sink sink)throws Exception{
        long[] t=prefix(true);
        TokenMetadata.Mapping m=tokens.map(t);
        if(m.chars.length==0)return false;
        float[][][] features;
        if(cache!=null){cachedStep(cache,new long[]{tokens.comma},StageStats.COMMA);features=cache.allFeatures();}
        else features=predict(t,encoded,StageStats.COMMA).features;
        if(features[0].length!=t.length)throw new IllegalStateException("decoder feature alignment");
        long unitStart=System.nanoTime();
        long[] u;
        int nextDuration;
        try(OnnxTensor f=OnnxTensor.createTensor(env,features);OnnxTensor ch=longs(m.chars);
            OnnxTensor len=longs(m.lengths);OrtSession.Result r=units.run(Map.of("features",f,"chars",ch,"char_lens",len))){
            long[] d=((long[][])value(r,"durations"))[0];
            if(durationStart>d.length)throw new IllegalStateException("duration prefix shrank");
            int total=StreamingPolicy.offset(d,d.length),start=StreamingPolicy.offset(d,durationStart);
            if(total-start==0)return flush;
            if(total-start<(flush?1:50))return false;
            u=StreamingPolicy.units(((float[][][])value(r,"logits"))[0],d,durationStart);
            nextDuration=flush?d.length:Math.max(0,d.length-1);
        }finally{stats.add(StageStats.UNITS,unitStart,m.chars.length);}
        long vocoderStart=System.nanoTime();
        float[] waveform;
        try{waveform=vocoder.decode(u);}finally{stats.add(StageStats.VOCODER,vocoderStart,u.length);}
        if(waveform.length==0)throw new IllegalStateException("empty synthesized output");
        sink.audio(waveform);
        durationStart=nextDuration;
        return true;
    }
    @Override public void close()throws Exception{
        Exception first=null;
        for(AutoCloseable c:new AutoCloseable[]{vocoder,units,decoder,encoder})if(c!=null)try{c.close();}catch(Exception e){if(first==null)first=e;}
        if(first!=null)throw first;
    }
}
