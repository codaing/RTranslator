package nie.translator.seamless;

/** Opt-in experimental rollover. Never treat elapsed time alone as source consumption. */
final class ContextBoundary {
    static final int MIN_CONTEXT_FRAMES=800, QUIET_SAMPLES=9600, BLOCK=160;
    private long energy;
    private int blockSamples, quietSamples;
    void observe(short[] pcm){
        for(short s:pcm){
            energy+=(long)s*s;
            if(++blockSamples==BLOCK){
                if(energy<=BLOCK*250L*250L)quietSamples=Math.min(QUIET_SAMPLES,quietSamples+BLOCK);
                else quietSamples=0;
                energy=0;blockSamples=0;
            }
        }
    }
    boolean candidate(boolean enabled,int frames,int tokens,boolean eos,float probability){
        return enabled&&frames>=MIN_CONTEXT_FRAMES&&tokens>=2&&quietSamples>=QUIET_SAMPLES&&eos&&probability>=0.5f;
    }
    void reset(){quietSamples=0;energy=0;blockSamples=0;}
}
