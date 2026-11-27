package nie.translator.seamless;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.media.AudioManager;
import android.net.Uri;
import android.os.Bundle;
import android.view.WindowManager;
import android.widget.*;
import java.io.*;
import java.util.concurrent.*;

public final class MainActivity extends Activity {
    private final ExecutorService worker=Executors.newSingleThreadExecutor();
    private TextView status;
    private Button start,load,stop;
    private volatile LiveAudio live;
    private volatile boolean closed, busy, stopRequested;
    private File modelDir(){return new File(getFilesDir(),"models");}
    @Override public void onCreate(Bundle state){
        super.onCreate(state);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        LinearLayout panel=new LinearLayout(this);panel.setOrientation(LinearLayout.VERTICAL);panel.setPadding(32,48,32,32);
        TextView title=new TextView(this);title.setText("SeamlessStreaming • Mandarin → English");title.setTextSize(24);panel.addView(title);
        status=new TextView(this);status.setText("EXPERIMENTAL — model conversion and full speech translation not validated. Headphones required. 60-second / 100-token context limit. Other languages are not filtered.");panel.addView(status);
        load=new Button(this);load.setText("Import model ZIP");panel.addView(load);
        start=new Button(this);start.setText("Start continuous interpretation test");panel.addView(start);
        stop=new Button(this);stop.setText("Stop and flush");panel.addView(stop);
        load.setOnClickListener(v->{Intent i=new Intent(Intent.ACTION_OPEN_DOCUMENT);i.setType("*/*");i.addCategory(Intent.CATEGORY_OPENABLE);startActivityForResult(i,17);});
        start.setOnClickListener(v->{
            if(checkSelfPermission(Manifest.permission.RECORD_AUDIO)!=PackageManager.PERMISSION_GRANTED){requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO},18);return;}
            startLive();
        });
        stop.setOnClickListener(v->stopLive());
        setContentView(panel);buttons();
    }
    private void buttons(){load.setEnabled(!busy);start.setEnabled(!busy&&ModelPack.present(modelDir()));stop.setEnabled(busy);}
    private void show(String s){runOnUiThread(()->{if(!closed)status.setText(s);});}
    private void done(){runOnUiThread(()->{busy=false;if(!closed)buttons();});}
    @Override protected void onActivityResult(int request,int result,Intent data){
        super.onActivityResult(request,result,data);
        if(request!=17||result!=RESULT_OK||data==null||data.getData()==null||busy)return;
        Uri uri=data.getData();busy=true;buttons();show("Importing model pack…");
        worker.execute(()->{
            try(InputStream in=getContentResolver().openInputStream(uri)){
                if(in==null)throw new IOException("Cannot open file");
                ModelPack.importZip(in,modelDir());show("Models imported. Integration is unvalidated; connect headphones before starting.");
            }catch(Exception e){show("Import failed: "+e);}
            finally{done();}
        });
    }
    @Override public void onRequestPermissionsResult(int code,String[] permissions,int[] grants){
        super.onRequestPermissionsResult(code,permissions,grants);
        if(code==18&&grants.length>0&&grants[0]==PackageManager.PERMISSION_GRANTED&&!busy)startLive();
    }
    private void startLive(){
        if(busy)return;
        busy=true;stopRequested=false;buttons();show("Loading four model sessions…");
        worker.execute(()->{
            try(StreamingEngine engine=new StreamingEngine(modelDir())){
                if(stopRequested||closed)return;
                LiveAudio audio=new LiveAudio(getSystemService(AudioManager.class),this::show);live=audio;
                if(stopRequested||closed){audio.close();return;}
                audio.run(engine);show("Stopped.");
            }catch(Exception|OutOfMemoryError e){show("Stopped: "+e+(e.getCause()==null?"":" — "+e.getCause()));}
            finally{live=null;done();}
        });
    }
    private void stopLive(){stopRequested=true;LiveAudio audio=live;if(audio!=null)audio.close();}
    @Override protected void onStop(){stopLive();super.onStop();}
    @Override protected void onDestroy(){closed=true;stopLive();worker.shutdown();super.onDestroy();}
}
