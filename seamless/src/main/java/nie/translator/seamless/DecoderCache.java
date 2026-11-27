package nie.translator.seamless;

import ai.onnxruntime.*;
import java.io.*;
import java.nio.*;
import java.util.*;
import org.json.*;

/** One source update owns its cache. Never reuse source-conditioned states across updates. */
final class DecoderCache implements AutoCloseable {
    static final String GRAPH="decoder.incremental.fp16.onnx", META="decoder.incremental.meta.json";
    static final String SOURCE="decoder.source-projections.onnx", STEP="decoder.cached-step.onnx";
    static final class Layout {
        final int heads,dim,layers;
        final String[] past,present;
        Layout(File dir)throws Exception{
            File f=new File(dir,META);
            if(f.length()>65536)throw new IOException("cache metadata too large");
            ByteArrayOutputStream bytes=new ByteArrayOutputStream();
            try(InputStream in=new FileInputStream(f)){byte[] b=new byte[4096];int n;while((n=in.read(b))!=-1)bytes.write(b,0,n);}
            JSONObject j=new JSONObject(bytes.toString("UTF-8"));
            heads=j.getInt("heads");dim=j.getInt("head_dim");layers=j.getInt("layers");
            if(heads<1||heads>64||dim<1||dim>256||layers<1||layers>48)
                throw new IOException("invalid cache dimensions");
            past=new String[2*layers];present=new String[2*layers];
            JSONArray p=j.getJSONArray("past_names"),q=j.getJSONArray("present_names");
            if(p.length()!=past.length||q.length()!=present.length)throw new IOException("invalid cache count");
            for(int i=0;i<layers;i++)for(int k=0;k<2;k++){
                String suffix=String.format(Locale.US,"%s_%02d",k==0?"k":"v",i);
                int n=2*i+k;past[n]="past_"+suffix;present[n]="present_"+suffix;
                if(!past[n].equals(p.getString(n))||!present[n].equals(q.getString(n)))
                    throw new IOException("invalid cache names");
            }
        }
        void check(OrtSession s){check(s,null);}
        void check(OrtSession s,OrtSession projection){
            Set<String> in=new HashSet<>(Arrays.asList(past));in.addAll(Arrays.asList("tokens","position"));
            if(projection==null)in.add("encoded");
            else{
                if(!projection.getInputNames().equals(Collections.singleton("encoded"))||
                   projection.getOutputNames().isEmpty())throw new IllegalStateException("projection interface mismatch");
                for(String name:projection.getOutputNames())
                    if(!in.add(name))throw new IllegalStateException("projection input collision");
                if(s.getInputNames().contains("encoded"))in.add("encoded");
            }
            Set<String> out=new HashSet<>(Arrays.asList(present));out.addAll(Arrays.asList("logits","features","p_choose"));
            if(!in.equals(s.getInputNames())||!out.equals(s.getOutputNames()))
                throw new IllegalStateException("incremental decoder interface mismatch");
        }
    }
    private final OrtEnvironment env;
    private final OrtSession session;
    private final Layout layout;
    private final OnnxTensor encoded;
    private final OnnxTensor empty;
    private final Map<String,OnnxTensor> projected=new HashMap<>();
    private OrtSession.Result previous;
    private int position;
    private final List<float[]> features=new ArrayList<>();
    DecoderCache(OrtEnvironment env,OrtSession session,Layout layout,float[][][] source)throws OrtException{
        this(env,session,layout,source,null);
    }
    DecoderCache(OrtEnvironment env,OrtSession session,Layout layout,float[][][] source,
                 OrtSession.Result projections)throws OrtException{
        this.env=env;this.session=session;this.layout=layout;
        if(projections!=null)for(Map.Entry<String,OnnxValue> entry:projections){
            if(!(entry.getValue() instanceof OnnxTensor))throw new IllegalStateException("projection must be a tensor");
            projected.put(entry.getKey(),(OnnxTensor)entry.getValue());
        }
        // Projection result is owned by the enclosing source update, not this cache.
        encoded=session.getInputNames().contains("encoded")?OnnxTensor.createTensor(env,source):null;
        try{
            empty=OnnxTensor.createTensor(env,ByteBuffer.allocateDirect(0).order(ByteOrder.nativeOrder()).asFloatBuffer(),
                new long[]{1,layout.heads,0,layout.dim});
        }catch(OrtException|RuntimeException|OutOfMemoryError e){if(encoded!=null)encoded.close();throw e;}
    }
    StreamingEngine.Prediction step(long[] ids)throws Exception{
        if(ids.length==0)throw new IllegalArgumentException("empty decoder step");
        Map<String,OnnxTensor> inputs=new HashMap<>();
        try(OnnxTensor t=OnnxTensor.createTensor(env,LongBuffer.wrap(ids),new long[]{1,ids.length});
            OnnxTensor p=OnnxTensor.createTensor(env,LongBuffer.wrap(new long[]{position}),new long[]{1})){
            inputs.put("tokens",t);inputs.put("position",p);
            if(encoded!=null)inputs.put("encoded",encoded);
            inputs.putAll(projected);
            for(int i=0;i<layout.past.length;i++){
                OnnxTensor cache=previous==null?empty:(OnnxTensor)previous.get(layout.present[i]).orElseThrow();
                inputs.put(layout.past[i],cache);
            }
            OrtSession.Result next=session.run(inputs);
            try{
                float[][][] logits=(float[][][])next.get("logits").orElseThrow().getValue();
                float[][][] f=(float[][][])next.get("features").orElseThrow().getValue();
                float prob=StreamingPolicy.probability((float[][][])next.get("p_choose").orElseThrow().getValue());
                if(f.length!=1||f[0].length!=ids.length)throw new IllegalStateException("incremental feature alignment");
                for(int i=0;i<layout.present.length;i++){
                    OnnxTensor cache=(OnnxTensor)next.get(layout.present[i]).orElseThrow();
                    long[] shape=cache.getInfo().getShape();
                    if(!Arrays.equals(shape,new long[]{1,layout.heads,position+ids.length,layout.dim}))
                        throw new IllegalStateException("incremental cache shape mismatch");
                }
                int token=StreamingPolicy.argmax(logits[0][logits[0].length-1]);
                Collections.addAll(features,f[0]);position+=ids.length;
                if(previous!=null)previous.close();
                previous=next;next=null;
                return new StreamingEngine.Prediction(token,prob,f);
            }finally{if(next!=null)next.close();}
        }
    }
    float[][][] allFeatures(){return new float[][][]{features.toArray(new float[0][])};}
    @Override public void close(){
        try{if(previous!=null)previous.close();}finally{try{empty.close();}finally{if(encoded!=null)encoded.close();}}
    }
}
