package nie.translator.seamless;
import java.io.*;
import java.util.*;
import java.util.zip.*;

/** Flat ZIP only, external ONNX data must sit beside the graph. */
final class ModelPack {
    static final String[] REQUIRED={"encoder.int8.onnx","decoder.int8.onnx","units.onnx","vocoder.onnx","tokenizer.json"};
    static boolean present(File dir){
        for(String f:REQUIRED)if(!new File(dir,f).isFile()||new File(dir,f).length()==0)return false;
        return true;
    }
    static void clear(File dir)throws IOException{
        if(!dir.exists())return;
        File[] files=dir.listFiles();
        if(files!=null)for(File f:files){if(f.isDirectory())clear(f);else if(!f.delete())throw new IOException("cannot remove "+f);}
        if(!dir.delete())throw new IOException("cannot remove "+dir);
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
            // Keep previous model folder until staging succeeds.
            File backup=new File(dir.getParentFile(),"models-old");
            clear(backup);
            if(dir.exists()&&!dir.renameTo(backup))throw new IOException("cannot stage previous models");
            if(!tmp.renameTo(dir)){backup.renameTo(dir);throw new IOException("cannot activate model folder");}
            clear(backup);
        }finally{clear(tmp);}
    }
}
