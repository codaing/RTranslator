package nie.translator.seamless;

import java.util.Arrays;

/** Pure-Java policy functions independently tested without checkpoints. */
public final class StreamingPolicy {
    private StreamingPolicy(){}
    public static int argmax(float[] scores){
        if(scores.length==0)throw new IllegalArgumentException("empty logits");
        int best=0;
        for(int i=0;i<scores.length;i++){
            if(!Float.isFinite(scores[i]))throw new IllegalStateException("non-finite logits");
            if(scores[i]>scores[best])best=i;
        }
        return best;
    }
    public static float probability(float[][][] choose){
        float min=1;
        if(choose.length==0)throw new IllegalStateException("empty decision");
        for(float[][] head:choose){
            float[] last=head[head.length-1];
            float v=last[last.length-1];
            if(!Float.isFinite(v)||v<0||v>1)throw new IllegalStateException("invalid decision");
            min=Math.min(min,v);
        }
        return min;
    }
    public static boolean write(float p,int token,int eos,boolean finished){
        return token!=eos&&(finished||p>=0.5f);
    }
    public static int offset(long[] durations,int start){
        if(start<0||start>durations.length)throw new IllegalStateException("duration context changed");
        long sum=0;
        for(int i=0;i<start;i++){if(durations[i]<0)throw new IllegalStateException("negative duration");sum+=durations[i];}
        if(sum>Integer.MAX_VALUE)throw new IllegalStateException("duration overflow");
        return (int)sum;
    }
    public static long[] units(float[][] logits,long[] durations,int start){
        int total=offset(durations,durations.length),begin=offset(durations,start);
        if(total!=logits.length)throw new IllegalStateException("unit duration mismatch");
        long[] out=new long[total-begin];
        for(int i=begin;i<total;i++){
            int id=argmax(logits[i]);if(id==1||id==2)id=5;
            id-=4;
            if(id<0||id>=10000)throw new IllegalStateException("unsupported acoustic token "+id);
            out[i-begin]=id;
        }
        return out;
    }
}
