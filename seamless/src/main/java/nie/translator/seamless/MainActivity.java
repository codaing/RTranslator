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
    private Button start,load,stop,benchmark,copyReport,probeGpu;
    private CheckBox rollover, xnnpack, nnapi, quietCpu;
    private Spinner gpuProfile;
    private volatile LiveAudio live;
    private volatile boolean closed, busy, stopRequested;
    private volatile String lastReport;
    private File modelDir(){return new File(getFilesDir(),"models");}
    @Override public void onCreate(Bundle state){
        super.onCreate(state);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        LinearLayout panel=new LinearLayout(this);panel.setOrientation(LinearLayout.VERTICAL);panel.setPadding(32,48,32,32);
        TextView title=new TextView(this);title.setText("SeamlessStreaming • Mandarin → English");title.setTextSize(24);panel.addView(title);
        status=new TextView(this);status.setText("EXPERIMENTAL — model conversion and full speech translation not validated. Headphones required. 60-second / 100-token safety limit. Optional quiet-boundary reset is unvalidated. Other languages are not filtered.");panel.addView(status);
        rollover=new CheckBox(this);rollover.setText("Experimental adaptive context reset (may omit or repeat speech)");rollover.setChecked(false);panel.addView(rollover);
        xnnpack=new CheckBox(this);xnnpack.setText("Try XNNPACK on acoustic-unit decoder (experimental)");xnnpack.setChecked(false);panel.addView(xnnpack);
        nnapi=new CheckBox(this);nnapi.setText("Try Android NNAPI on the speech encoder (experimental)");nnapi.setChecked(false);panel.addView(nnapi);
        quietCpu=new CheckBox(this);quietCpu.setText("Quiet CPU pools (experimental: 4 threads, no idle spinning)");quietCpu.setChecked(false);panel.addView(quietCpu);
        TextView gpuLabel=new TextView(this);gpuLabel.setText("Offline benchmark only: WebGPU profile (experimental)");panel.addView(gpuLabel);
        gpuProfile=new Spinner(this);
        ArrayAdapter<GpuTuning> gpuAdapter=new ArrayAdapter<>(this,android.R.layout.simple_spinner_item,GpuTuning.values());
        gpuAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        gpuProfile.setAdapter(gpuAdapter);gpuProfile.setSelection(0);panel.addView(gpuProfile);
        probeGpu=new Button(this);probeGpu.setText("Test WebGPU FP32 / FP16 (no model needed)");panel.addView(probeGpu);
        probeGpu.setOnClickListener(v->runGpuProbe());
        load=new Button(this);load.setText("Import model ZIP");panel.addView(load);
        start=new Button(this);start.setText("Start continuous interpretation test");panel.addView(start);
        benchmark=new Button(this);benchmark.setText("Benchmark local 16 kHz mono PCM WAV");panel.addView(benchmark);
        copyReport=new Button(this);copyReport.setText("Copy benchmark report");panel.addView(copyReport);
        stop=new Button(this);stop.setText("Stop and flush / cancel benchmark");panel.addView(stop);
        benchmark.setOnClickListener(v->{Intent i=new Intent(Intent.ACTION_OPEN_DOCUMENT);i.setType("*/*");i.addCategory(Intent.CATEGORY_OPENABLE);startActivityForResult(i,19);});
        copyReport.setOnClickListener(v->{
            if(lastReport!=null)getSystemService(android.content.ClipboardManager.class).setPrimaryClip(android.content.ClipData.newPlainText("Seamless benchmark",lastReport));
        });
        load.setOnClickListener(v->{Intent i=new Intent(Intent.ACTION_OPEN_DOCUMENT);i.setType("*/*");i.addCategory(Intent.CATEGORY_OPENABLE);startActivityForResult(i,17);});
        start.setOnClickListener(v->{
            if(checkSelfPermission(Manifest.permission.RECORD_AUDIO)!=PackageManager.PERMISSION_GRANTED){requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO},18);return;}
            startLive();
        });
        stop.setOnClickListener(v->stopLive());
        ScrollView scroll=new ScrollView(this);scroll.addView(panel);setContentView(scroll);buttons();
    }
    private void buttons(){load.setEnabled(!busy);start.setEnabled(!busy&&ModelPack.present(modelDir()));stop.setEnabled(busy);rollover.setEnabled(!busy);xnnpack.setEnabled(!busy);nnapi.setEnabled(!busy);quietCpu.setEnabled(!busy);gpuProfile.setEnabled(!busy);probeGpu.setEnabled(!busy);benchmark.setEnabled(!busy&&ModelPack.present(modelDir()));copyReport.setEnabled(!busy&&lastReport!=null);}
    private void show(String s){runOnUiThread(()->{if(!closed)status.setText(s);});}
    private void done(){runOnUiThread(()->{busy=false;if(!closed)buttons();});}
    @Override protected void onActivityResult(int request,int result,Intent data){
        super.onActivityResult(request,result,data);
        if(result!=RESULT_OK||data==null||data.getData()==null||busy)return;
        if(request==19){runBenchmark(data.getData());return;}
        if(request!=17)return;
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
        final boolean adaptive=rollover.isChecked();
        final boolean useXnnpack=xnnpack.isChecked(),useNnapi=nnapi.isChecked(),useQuiet=quietCpu.isChecked();
        busy=true;stopRequested=false;buttons();show("Loading four model sessions…");
        worker.execute(()->{
            try(StreamingEngine engine=new StreamingEngine(modelDir(),adaptive,useXnnpack,useNnapi,useQuiet)){
                if(stopRequested||closed)return;
                LiveAudio audio=new LiveAudio(getSystemService(AudioManager.class),this::show);live=audio;
                if(stopRequested||closed){audio.close();return;}
                audio.run(engine);
            }catch(Exception|OutOfMemoryError e){show("Stopped: "+e+(e.getCause()==null?"":" — "+e.getCause()));}
            finally{live=null;done();}
        });
    }
    private void runGpuProbe(){
        if(busy)return;
        busy=true;stopRequested=false;lastReport=null;buttons();show("Testing GPU-only matrix multiplication…");
        worker.execute(()->{
            StringBuilder report=new StringBuilder("WebGPU capability probe (CPU fallback disabled)\nDevice "+android.os.Build.MODEL);
            try{
                report.append("\nAndroid ").append(android.os.Build.VERSION.RELEASE)
                      .append(" | SDK ").append(android.os.Build.VERSION.SDK_INT);
                report.append("\n").append(VulkanProbe.run());
                if(!stopRequested)report.append("\n").append(VulkanProbe.matmul(1));
                if(!stopRequested)report.append("\n").append(VulkanProbe.matmul(32));
                boolean fp32=false;
                java.util.Map<String,String> options=java.util.Collections.emptyMap();
                boolean optimized=false;
                long t=android.os.SystemClock.elapsedRealtime();
                try{GpuTuning.probe(this,false);fp32=true;report.append("\nWebGPU FP32 default: PASS");}
                catch(Exception e){report.append("\nWebGPU FP32 default: FAILED ").append(e);}
                report.append(" (").append(android.os.SystemClock.elapsedRealtime()-t).append(" ms)");
                if(!fp32&&!stopRequested){
                    // Default-owned context only; nonzero deviceId requires external Dawn handles.
                    // Diagnostic only: eliminate buffer pooling and request full validation.
                    options=java.util.Map.of("deviceId","0","validationMode","full",
                        "storageBufferCacheMode","disabled","uniformBufferCacheMode","disabled",
                        "queryResolveBufferCacheMode","disabled","defaultBufferCacheMode","disabled");
                    optimized=true;t=android.os.SystemClock.elapsedRealtime();
                    try{GpuTuning.probe(this,false,options,true);fp32=true;report.append("\nWebGPU FP32 alternate: PASS");}
                    catch(Exception e){report.append("\nWebGPU FP32 alternate: FAILED ").append(e);}
                    report.append(" (").append(android.os.SystemClock.elapsedRealtime()-t).append(" ms)");
                }
                if(fp32&&!stopRequested){
                    t=android.os.SystemClock.elapsedRealtime();
                    try{GpuTuning.probe(this,true,options,optimized);report.append("\nWebGPU FP16: PASS");}
                    catch(ai.onnxruntime.OrtException e){
                        report.append(GpuTuning.missingShaderF16(e)?"\nFP16: shader-f16 unsupported":"\nFP16: FAILED");
                        report.append("\n").append(e.getMessage());
                    }
                    report.append(" (").append(android.os.SystemClock.elapsedRealtime()-t).append(" ms)");
                }else report.append("\nFP16: NOT TESTED");
            }catch(Exception|OutOfMemoryError e){report.append("\nProbe failed: ").append(e);}
            finally{
                lastReport=report.toString();show(lastReport);done();
            }
        });
    }
    private void runBenchmark(Uri uri){
        final boolean useXnnpack=xnnpack.isChecked(),useNnapi=nnapi.isChecked(),useQuiet=quietCpu.isChecked();
        final GpuTuning gpu=(GpuTuning)gpuProfile.getSelectedItem();
        busy=true;stopRequested=false;lastReport=null;buttons();show("Reading benchmark WAV…");
        worker.execute(()->{
            try(InputStream in=getContentResolver().openInputStream(uri)){
                if(in==null)throw new IOException("Cannot open WAV");
                if(gpu!=GpuTuning.CPU){
                    show("Checking GPU capability before loading model weights…");
                    GpuTuning.probe(this,false);
                    if(gpu.needsFp16Probe())GpuTuning.probe(this,true);
                }
                lastReport=OfflineBenchmark.run(modelDir(),in,()->stopRequested||closed,this::show,
                    useXnnpack,useNnapi,useQuiet,gpu);
                show(lastReport);
            }catch(Exception|OutOfMemoryError e){lastReport="Benchmark stopped: "+e;show(lastReport);}
            finally{done();}
        });
    }
    private void stopLive(){stopRequested=true;LiveAudio audio=live;if(audio!=null)audio.close();}
    @Override protected void onStop(){stopLive();super.onStop();}
    @Override protected void onDestroy(){closed=true;stopLive();worker.shutdown();super.onDestroy();}
}
