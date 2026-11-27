import nie.translator.seamless.StreamingFbank;
import java.io.*;
import java.nio.*;
import java.nio.file.*;
import java.util.*;

public class FbankCli {
    public static void main(String[] args) throws Exception {
        byte[] raw=Files.readAllBytes(Path.of(args[0]));
        if(raw.length%2!=0)throw new IOException("odd PCM bytes");
        ShortBuffer data=ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer();
        StreamingFbank f=new StreamingFbank();
        try(DataOutputStream out=new DataOutputStream(new BufferedOutputStream(new FileOutputStream(args[1])))){
            while(data.hasRemaining()){
                short[] pcm=new short[Math.min(5120,data.remaining())];data.get(pcm);
                for(float[] row:f.push(pcm))for(float x:row)out.writeInt(Integer.reverseBytes(Float.floatToRawIntBits(x)));
            }
        }
    }
}
