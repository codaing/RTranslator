package nie.translator.seamless;

import java.io.*;
import java.util.Arrays;
import java.util.Locale;
import java.util.concurrent.CancellationException;
import java.util.function.BooleanSupplier;
import android.os.Debug;
import android.os.SystemClock;

/** Offline diagnostic; consumes source in the same 320 ms transport frames. */
final class OfflineBenchmark {
    static String run(File dir,InputStream in,BooleanSupplier cancelled,LiveAudio.Status status,
                      boolean xnnpackFp32Stages,boolean nnapiEncoder,boolean quietCpu)throws Exception{
        short[] samples=BenchmarkWav.read(in);
        if(cancelled.getAsBoolean())throw new CancellationException("Cancelled");
        status.report("Benchmark: loading real model sessions…");
        long start=SystemClock.elapsedRealtime(),load;
        final long[] generated={0},first={-1},firstSource={-1},peak={Debug.getPss()};
        final long[] consumed={0};
        final boolean[] eof={false},beforeEof={false};
        final double[] energy={0};
        long computeStart;
        int tokens;String decoderMode,stageReport;
        long pushNanos=0,pssNanos=0,closeStart;
        long pssStart;
        try(StreamingEngine engine=new StreamingEngine(dir,false,xnnpackFp32Stages,nnapiEncoder,quietCpu)){
            load=SystemClock.elapsedRealtime()-start;
            peak[0]=Math.max(peak[0],Debug.getPss());
            computeStart=SystemClock.elapsedRealtime();
            StreamingEngine.Sink sink=pcm->{
                generated[0]+=pcm.length;
                for(float f:pcm)energy[0]+=(double)f*f;
                if(first[0]<0){
                    first[0]=SystemClock.elapsedRealtime()-computeStart;
                    firstSource[0]=consumed[0];beforeEof[0]=!eof[0];
                }
            };
            for(int i=0;i<samples.length;i+=5120){
                if(cancelled.getAsBoolean())throw new CancellationException("Cancelled");
                int end=Math.min(samples.length,i+5120);
                consumed[0]=end;
                long pushStart=System.nanoTime();
                engine.push(Arrays.copyOfRange(samples,i,end),false,sink);
                pushNanos+=System.nanoTime()-pushStart;
                pssStart=System.nanoTime();
                peak[0]=Math.max(peak[0],Debug.getPss());
                pssNanos+=System.nanoTime()-pssStart;
                status.report(String.format(Locale.US,"Benchmark: %.2f / %.2f s source | generated %.2f s | sampled PSS %.0f MiB",end/16000.0,samples.length/16000.0,generated[0]/16000.0,peak[0]/1024.0));
            }
            if(cancelled.getAsBoolean())throw new CancellationException("Cancelled");
            eof[0]=true;
            long pushStart=System.nanoTime();
            engine.push(new short[0],true,sink);
            pushNanos+=System.nanoTime()-pushStart;
            pssStart=System.nanoTime();
            peak[0]=Math.max(peak[0],Debug.getPss());
            pssNanos+=System.nanoTime()-pssStart;
            tokens=engine.targetTokens();decoderMode=engine.decoderMode();
            stageReport=engine.stats.report();closeStart=System.nanoTime();
        }
        long elapsed=SystemClock.elapsedRealtime()-computeStart;
        double duration=samples.length/16000.0,rtf=elapsed/1000.0/duration;
        return String.format(Locale.US,"Offline WAV benchmark (experimental)\nDevice %s / Android %s\nDecoder %s\nNNAPI encoder requested: %s | XNNPACK units requested: %s | quiet CPU requested: %s\nInput %.2f s | generated %.2f s\nModel load %.2f s | processing %.2f s (includes final flush and session close)\nRTF %.3f (below 1 is faster than audio duration)\nFirst generated audio %s | source consumed %s | before flush %s\nSampled peak PSS %.0f MiB | tokens %d\nWaveform RMS %.6f\nNo microphone, playback, network or TTS used. This is not a translation accuracy or live headphone latency test.",
            android.os.Build.MODEL,android.os.Build.VERSION.RELEASE,decoderMode,nnapiEncoder,xnnpackFp32Stages,quietCpu,duration,generated[0]/16000.0,load/1000.0,elapsed/1000.0,rtf,
            first[0]<0?"none":String.format(Locale.US,"%.2f s",first[0]/1000.0),firstSource[0]<0?"none":String.format(Locale.US,"%.2f s",firstSource[0]/16000.0),beforeEof[0]?"yes":"no",
            peak[0]/1024.0,tokens,generated[0]>0?Math.sqrt(energy[0]/generated[0]):0)
            +"\n\n"+stageReport+String.format(Locale.US,
                "All push calls: %.3f s | PSS sampling: %.3f s | session close: %.3f s\nInput items: PCM samples / feature frames / decoder tokens / characters / acoustic units.",
                pushNanos/1e9,pssNanos/1e9,(System.nanoTime()-closeStart)/1e9);
    }
}
