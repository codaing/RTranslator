package nie.translator.seamless;

import java.io.*;
import org.junit.Test;
import static org.junit.Assert.*;

public class BenchmarkWavTest {
    private static byte[] wav(int rate,int channels,int samples)throws IOException{
        ByteArrayOutputStream bytes=new ByteArrayOutputStream();
        DataOutputStream d=new DataOutputStream(bytes);
        d.writeBytes("RIFF");d.writeInt(Integer.reverseBytes(36+2*samples));d.writeBytes("WAVEfmt ");
        d.writeInt(Integer.reverseBytes(16));d.writeShort(Short.reverseBytes((short)1));
        d.writeShort(Short.reverseBytes((short)channels));d.writeInt(Integer.reverseBytes(rate));d.writeInt(Integer.reverseBytes(rate*channels*2));
        d.writeShort(Short.reverseBytes((short)(channels*2)));d.writeShort(Short.reverseBytes((short)16));
        d.writeBytes("data");d.writeInt(Integer.reverseBytes(2*samples));
        for(int i=0;i<samples;i++)d.writeShort(Short.reverseBytes((short)(i-8000)));
        return bytes.toByteArray();
    }
    @Test public void readsExactSignedSamples()throws Exception{
        short[] s=BenchmarkWav.read(new ByteArrayInputStream(wav(16000,1,16000)));
        assertEquals(16000,s.length);assertEquals(-8000,s[0]);assertEquals(7999,s[15999]);
    }
    @Test public void rejectsWrongRateAndTruncation()throws Exception{
        try{BenchmarkWav.read(new ByteArrayInputStream(wav(44100,1,16000)));fail();}catch(IOException expected){}
        byte[] full=wav(16000,1,16000);
        try{BenchmarkWav.read(new ByteArrayInputStream(java.util.Arrays.copyOf(full,full.length-1)));fail();}catch(IOException expected){}
    }
}
