package nie.translator.seamless;
import org.junit.Test;
import static org.junit.Assert.*;
import java.util.*;

public class StreamingTest {
    @Test public void featuresDoNotDependOnTransportFrameBoundaries(){
        short[] pcm=new short[16000];
        for(int i=0;i<pcm.length;i++)pcm[i]=(short)(8000*Math.sin(2*Math.PI*440*i/16000));
        List<float[]> all=new StreamingFbank().push(pcm);
        StreamingFbank stream=new StreamingFbank();List<float[]> parts=new ArrayList<>();
        for(int i=0;i<pcm.length;i+=317)parts.addAll(stream.push(Arrays.copyOfRange(pcm,i,Math.min(i+317,pcm.length))));
        assertEquals(98,all.size());assertEquals(all.size(),parts.size());
        for(int i=0;i<all.size();i++){
            assertArrayEquals(all.get(i),parts.get(i),0f);
            for(float x:parts.get(i))assertTrue(Float.isFinite(x));
        }
    }
    @Test public void readWriteUsesMinimumHeadAndNoEarlyEos(){
        assertEquals(0.4f,StreamingPolicy.probability(new float[][][]{{{0.9f,0.7f}},{{0.9f,0.4f}}}),0f);
        assertFalse(StreamingPolicy.write(0.4f,7,2,false));
        assertTrue(StreamingPolicy.write(0.5f,7,2,false));
        assertFalse(StreamingPolicy.write(0.9f,2,2,false));
        assertTrue(StreamingPolicy.write(0.1f,7,2,true));
    }
    @Test public void unitOffsetsAndControlSymbols(){
        float[][] scores=new float[4][8];scores[0][4]=1;scores[1][5]=1;scores[2][2]=1;scores[3][7]=1;
        assertArrayEquals(new long[]{1,1,3},StreamingPolicy.units(scores,new long[]{1,3},1));
    }
    @Test public void adaptiveRolloverRequiresQuietAndModelEos(){
        ContextBoundary c=new ContextBoundary();
        assertFalse(c.candidate(true,900,20,true,1f));
        c.observe(new short[ContextBoundary.QUIET_SAMPLES]);
        assertFalse(c.candidate(false,900,20,true,1f));
        assertFalse(c.candidate(true,799,20,true,1f));
        assertFalse(c.candidate(true,900,20,false,1f));
        assertFalse(c.candidate(true,900,20,true,0.49f));
        assertTrue(c.candidate(true,900,20,true,0.5f));
        short[] speech=new short[160];Arrays.fill(speech,(short)1000);c.observe(speech);
        assertFalse(c.candidate(true,900,20,true,1f));
        c.observe(new short[9600]);c.reset();
        assertFalse(c.candidate(true,900,20,true,1f));
    }
}
