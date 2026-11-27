package nie.translator.seamless;

import org.json.JSONArray;
import org.json.JSONObject;
import java.io.File;
import java.io.FileInputStream;
import java.io.ByteArrayOutputStream;
import java.util.ArrayList;
import java.util.List;

final class TokenMetadata {
    final long[] prefix;
    final int eos,pad,unk,comma;
    private final int[][] chars;
    private final boolean[] punctuation,space;
    TokenMetadata(File f)throws Exception{
        if(f.length()>64L*1024*1024)throw new IllegalArgumentException("token table too large");
        ByteArrayOutputStream bytes=new ByteArrayOutputStream();
        try(FileInputStream in=new FileInputStream(f)){
            byte[] buf=new byte[65536];int n;while((n=in.read(buf))!=-1)bytes.write(buf,0,n);
        }
        JSONObject meta=new JSONObject(bytes.toString("UTF-8"));
        JSONArray pre=meta.getJSONArray("prefix");prefix=new long[pre.length()];
        for(int i=0;i<prefix.length;i++)prefix[i]=pre.getLong(i);
        if(prefix.length!=2)throw new IllegalArgumentException("expected two-token English prefix");
        eos=meta.getInt("eos");pad=meta.getInt("pad");unk=meta.getInt("unk");comma=meta.getInt("comma");
        JSONArray table=meta.getJSONArray("table");
        chars=new int[table.length()][];punctuation=new boolean[table.length()];space=new boolean[table.length()];
        for(int i=0;i<table.length();i++){
            JSONArray row=table.getJSONArray(i),ids=row.getJSONArray(0);
            chars[i]=new int[ids.length()];for(int j=0;j<ids.length();j++)chars[i][j]=ids.getInt(j);
            punctuation[i]=row.getBoolean(1);space[i]=row.getBoolean(2);
        }
    }
    static final class Mapping {
        final long[] chars,lengths;
        Mapping(long[] c,long[] l){chars=c;lengths=l;}
    }
    Mapping map(long[] tokens){
        List<Long> out=new ArrayList<>();long[] lengths=new long[tokens.length];
        int end=tokens.length;
        for(int i=2;i<tokens.length;i++)if(tokens[i]==eos||tokens[i]==pad){end=i;break;}
        for(int i=2;i<end;i++){
            int id=Math.toIntExact(tokens[i]);
            if(id<0||id>=chars.length)throw new IllegalArgumentException("token out of range");
            int len=chars[id].length;
            if(id!=unk){
                if(i+1<end&&punctuation[id]&&space[(int)tokens[i+1]])len++;
                else if(i>2&&punctuation[(int)tokens[i-1]]&&space[id])len--;
            }
            // Upstream places lengths between a leading and trailing zero.
            lengths[i-1]=len;
            for(int ch:chars[id])out.add((long)ch);
        }
        long[] result=new long[out.size()];for(int i=0;i<result.length;i++)result[i]=out.get(i);
        if(StreamingPolicy.offset(lengths,lengths.length)!=result.length)throw new IllegalStateException("character alignment mismatch");
        return new Mapping(result,lengths);
    }
}
