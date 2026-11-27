package nie.translator.seamless;
import java.io.*;
import java.util.*;
import java.util.zip.*;
import java.security.MessageDigest;
import org.json.JSONObject;

/** Flat ZIP only, external ONNX data must sit beside the graph. */
final class ModelPack {
    static final String[] REQUIRED={"encoder.int8.onnx","decoder.int8.onnx","units.onnx","vocoder.onnx","tokenizer.json"};
    static boolean present(File dir){
        if(!new File(dir,"sha256.json").isFile())return false;
        for(String f:REQUIRED)if(!new File(dir,f).isFile()||new File(dir,f).length()==0)return false;
        return true;
    }
    static void clear(File dir)throws IOException{
        if(!dir.exists())return;
        File[] files=dir.listFiles();
        if(files!=null)for(File f:files){if(f.isDirectory())clear(f);else if(!f.delete())throw new IOException("cannot remove "+f);}
        if(!dir.delete())throw new IOException("cannot remove "+dir);
    }
    /** Integrity check detects corruption, not malicious substitution of a manifest. */
    static void verify(File dir)throws Exception{
        File manifest=new File(dir,"sha256.json");
        if(manifest.length()>8L*1024*1024)throw new IOException("hash manifest too large");
        ByteArrayOutputStream bytes=new ByteArrayOutputStream();
        try(InputStream in=new FileInputStream(manifest)){byte[] b=new byte[65536];int n;while((n=in.read(b))!=-1)bytes.write(b,0,n);}
        JSONObject hashes=new JSONObject(bytes.toString("UTF-8"));
        File[] files=dir.listFiles();
        if(files==null)throw new IOException("cannot list model directory");
        int count=0;
        for(File f:files){
            if(f.getName().equals("sha256.json"))continue;
            if(!f.isFile())throw new IOException("unexpected model directory");
            String expected=hashes.getString(f.getName());
            if(!expected.matches("[a-f0-9]{64}"))throw new IOException("invalid SHA256 entry");
            MessageDigest digest=MessageDigest.getInstance("SHA-256");
            try(InputStream in=new BufferedInputStream(new FileInputStream(f))){
                byte[] b=new byte[1024*1024];int n;while((n=in.read(b))!=-1)digest.update(b,0,n);
            }
            StringBuilder actual=new StringBuilder();
            for(byte b:digest.digest())actual.append(String.format(Locale.US,"%02x",b&255));
            if(!expected.equals(actual.toString()))throw new IOException("SHA256 mismatch: "+f.getName());
            count++;
        }
        if(count!=hashes.length())throw new IOException("missing manifest file");
    }
    static void importZip(InputStream in,File dir)throws Exception{
        File tmp=new File(dir.getParentFile(),"models-import");
        clear(tmp);
        if(!tmp.mkdir())throw new IOException("cannot create model staging directory");
        Set<String> names=new HashSet<>();long total=0;
        try(ZipInputStream zip=new ZipInputStream(in)){
            ZipEntry e;
            while((e=zip.getNextEntry())!=null){
                String name=e.getName();
                if(e.isDirectory()||!name.matches("[A-Za-z0-9_.:-]{1,240}")||name.equals(".")||name.equals("..")||!names.add(name)||names.size()>20000)
                    throw new IOException("invalid or duplicate ZIP entry: "+name);
                try(FileOutputStream out=new FileOutputStream(new File(tmp,name))){
                    byte[] buf=new byte[65536];int n;
                    while((n=zip.read(buf))!=-1){
                        total+=n;if(total>16L*1024*1024*1024)throw new IOException("model archive exceeds 16 GiB");
                        out.write(buf,0,n);
                    }
                }
            }
            if(!present(tmp))throw new IOException("archive missing required graphs or tokenizer.json");
            verify(tmp);
            // Keep previous model folder until staging succeeds.
            File backup=new File(dir.getParentFile(),"models-old");
            clear(backup);
            if(dir.exists()&&!dir.renameTo(backup))throw new IOException("cannot stage previous models");
            if(!tmp.renameTo(dir)){backup.renameTo(dir);throw new IOException("cannot activate model folder");}
            clear(backup);
        }finally{clear(tmp);}
    }
}
