package nie.translator.seamless;

import java.util.Locale;

/** Single-worker cumulative timing, including tensor marshaling. Nonoverlapping stages. */
final class StageStats {
    static final int FBANK=0, ENCODER=1, PREFILL=2, STEP=3, COMMA=4, UNITS=5, VOCODER=6, PROJECTION=7;
    private static final String[] NAMES={"Filterbank","Encoder","Decoder prefill","Decoder token steps",
        "Decoder synthesis comma","Acoustic units","Vocoder","Source projections"};
    private final long[] nanos=new long[NAMES.length],calls=new long[NAMES.length],items=new long[NAMES.length];
    void add(int stage,long started,long count){
        nanos[stage]+=System.nanoTime()-started;calls[stage]++;items[stage]+=count;
    }
    long calls(int stage){return calls[stage];}
    String report(){
        StringBuilder b=new StringBuilder("Stage timing (includes tensor conversion):\n");
        long total=0;
        for(int i=0;i<NAMES.length;i++){
            total+=nanos[i];
            b.append(String.format(Locale.US,"%s: %.3f s | calls %d | input items %d\n",
                NAMES[i],nanos[i]/1e9,calls[i],items[i]));
        }
        return b.append(String.format(Locale.US,"Timed stage sum: %.3f s\n",total/1e9)).toString();
    }
}
