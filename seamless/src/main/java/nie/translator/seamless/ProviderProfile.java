package nie.translator.seamless;

import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtException;
import android.util.JsonReader;
import android.util.JsonToken;
import java.io.*;
import java.util.*;

/** Optional ORT node-event tracing. Summaries are not device-driver placement evidence. */
final class ProviderProfile {
    private final File directory;
    private final Map<String,OrtSession> sessions=new LinkedHashMap<>();
    private String report;
    ProviderProfile(File parent)throws IOException{
        directory=new File(parent,"provider-profiles");
        if(!directory.isDirectory()&&!directory.mkdirs())throw new IOException("Cannot create app-private profile directory");
    }
    void configure(OrtSession.SessionOptions options,String name)throws OrtException{
        options.enableProfiling(new File(directory,name+"-"+UUID.randomUUID()).getAbsolutePath());
    }
    void register(String name,OrtSession session){sessions.put(name,session);}
    String finish(){
        if(report!=null)return report;
        StringBuilder out=new StringBuilder("ORT provider node-event summary (summed kernel-event duration; not wall time or proof of TPU):");
        for(Map.Entry<String,OrtSession> entry:sessions.entrySet()){
            out.append("\n").append(entry.getKey()).append(": ");
            File file=null;
            try{
                file=new File(entry.getValue().endProfiling());
                Map<String,long[]> counts=summarize(file);
                if(counts.isEmpty())out.append("no provider-tagged node events; placement unverified");
                for(Map.Entry<String,long[]> row:counts.entrySet()){
                    out.append(String.format(Locale.US,"%s %d events / %.3f s; ",row.getKey(),row.getValue()[0],row.getValue()[1]/1e6));
                }
            }catch(Exception e){out.append("trace unavailable: ").append(e.getMessage());}
            finally{if(file!=null&&directory.equals(file.getParentFile()))file.delete();}
        }
        sessions.clear();
        report=out.toString();
        return report;
    }
    static Map<String,long[]> summarize(File file)throws IOException{
        Map<String,long[]> counts=new TreeMap<>();
        try(JsonReader r=new JsonReader(new BufferedReader(new FileReader(file)))){
            r.beginArray();
            while(r.hasNext()){
                String category="",name="",provider="";
                long duration=0;
                r.beginObject();
                while(r.hasNext()){
                    String key=r.nextName();
                    switch(key){
                        case "cat":category=r.nextString();break;
                        case "name":name=r.nextString();break;
                        case "dur":duration=r.nextLong();break;
                        case "args":
                            if(r.peek()!=JsonToken.BEGIN_OBJECT){r.skipValue();break;}
                            r.beginObject();
                            while(r.hasNext()){
                                if(r.nextName().equals("provider")&&r.peek()==JsonToken.STRING)provider=r.nextString();
                                else r.skipValue();
                            }
                            r.endObject();break;
                        default:r.skipValue();
                    }
                }
                r.endObject();
                if(category.equals("Node")&&name.endsWith("_kernel_time")&&!provider.isEmpty()&&duration>=0){
                    long[] c=counts.computeIfAbsent(provider,k->new long[2]);
                    c[0]++;c[1]+=duration;
                }
            }
            r.endArray();
        }
        return counts;
    }
}
