package nie.translator.seamless;

import java.io.*;
import java.nio.charset.StandardCharsets;

/** Strict bounded PCM WAV parser; no implicit resampling or partial input. */
final class BenchmarkWav {
    static final int MAX_SAMPLES=16000*30;
    private static long u32(DataInputStream in)throws IOException{return Integer.toUnsignedLong(Integer.reverseBytes(in.readInt()));}
    private static int u16(DataInputStream in)throws IOException{return Short.toUnsignedInt(Short.reverseBytes(in.readShort()));}
    private static String tag(DataInputStream in)throws IOException{byte[] b=new byte[4];in.readFully(b);return new String(b,StandardCharsets.US_ASCII);}
    private static void skip(DataInputStream in,long count)throws IOException{
        while(count>0){long n=in.skip(count);if(n==0){if(in.read()<0)throw new EOFException("truncated WAV");n=1;}count-=n;}
    }
    static short[] read(InputStream input)throws IOException{
        DataInputStream in=new DataInputStream(new BufferedInputStream(input));
        if(!tag(in).equals("RIFF"))throw new IOException("Expected RIFF WAV");
        long left=u32(in);
        if(left<4||left>4L*1024*1024)throw new IOException("WAV too large or invalid");
        if(!tag(in).equals("WAVE"))throw new IOException("Expected WAVE");
        left-=4;boolean format=false;
        while(left>=8){
            String kind=tag(in);long size=u32(in);left-=8;
            long padded=size+(size&1);
            if(padded>left)throw new IOException("Truncated WAV chunk");
            if(kind.equals("fmt ")){
                if(format||size<16)throw new IOException("Invalid WAV format chunk");
                int type=u16(in),channels=u16(in);long rate=u32(in),byteRate=u32(in);
                int align=u16(in),bits=u16(in);
                if(type!=1||channels!=1||rate!=16000||byteRate!=32000||align!=2||bits!=16)
                    throw new IOException("Use uncompressed mono 16 kHz signed 16-bit PCM WAV");
                skip(in,padded-16);format=true;
            }else if(kind.equals("data")){
                if(!format||size%2!=0||size<32000||size>2L*MAX_SAMPLES)
                    throw new IOException("Use a PCM WAV between 1 and 30 seconds");
                short[] samples=new short[(int)(size/2)];
                for(int i=0;i<samples.length;i++)samples[i]=Short.reverseBytes(in.readShort());
                return samples;
            }else skip(in,padded);
            left-=padded;
        }
        throw new IOException("Missing WAV data");
    }
}
