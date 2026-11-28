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
    private Button start,load,stop,benchmark,copyReport,probeGpu,compareGpu;
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
        probeGpu=new Button(this);probeGpu.setText("Test native Vulkan kernels (no model needed)");panel.addView(probeGpu);
        probeGpu.setOnClickListener(v->runGpuProbe());
        compareGpu=new Button(this);compareGpu.setText("Compare top Vulkan kernels (balanced A/B)");panel.addView(compareGpu);
        compareGpu.setOnClickListener(v->runGpuComparison());
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
    private void buttons(){load.setEnabled(!busy);start.setEnabled(!busy&&ModelPack.present(modelDir()));stop.setEnabled(busy);rollover.setEnabled(!busy);xnnpack.setEnabled(!busy);nnapi.setEnabled(!busy);quietCpu.setEnabled(!busy);gpuProfile.setEnabled(!busy);probeGpu.setEnabled(!busy);compareGpu.setEnabled(!busy);benchmark.setEnabled(!busy&&ModelPack.present(modelDir()));copyReport.setEnabled(!busy&&lastReport!=null);}
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
        busy=true;stopRequested=false;lastReport=null;buttons();show("Testing native Vulkan matrix kernels…");
        worker.execute(()->{
            StringBuilder report=new StringBuilder("Native Vulkan speed probe (synthetic, no CPU fallback)\nDevice "+android.os.Build.MODEL);
            try{
                report.append("\nAndroid ").append(android.os.Build.VERSION.RELEASE)
                      .append(" | SDK ").append(android.os.Build.VERSION.SDK_INT);
                report.append("\n").append(VulkanProbe.run());
                if(!stopRequested)report.append("\n").append(VulkanProbe.matmul(1));
                if(!stopRequested)report.append("\n").append(VulkanProbe.matmul(32));
                if(!stopRequested)report.append("\n").append(VulkanProbe.subgroup());
                for(int d:new int[]{1,4,8,16,32})if(!stopRequested)report.append("\n").append(VulkanProbe.gemv(d));
                for(int o:new int[]{4,8,16})if(!stopRequested)report.append("\n").append(VulkanProbe.fused(o));
                report.append("\nWebGPU retry deferred: repeated Pixel readback timeout.");
            }catch(Exception|OutOfMemoryError e){report.append("\nProbe failed: ").append(e);}
            finally{
                lastReport=report.toString();show(lastReport);done();
            }
        });
    }
    private void runGpuComparison(){
        if(busy)return;
        busy=true;stopRequested=false;lastReport=null;buttons();
        show("Running balanced Vulkan kernel comparison…");
        worker.execute(()->{
            StringBuilder report=new StringBuilder("Balanced Vulkan kernel comparison (synthetic; no CPU fallback)\nDevice "+android.os.Build.MODEL);
            try{
                report.append("\nAndroid ").append(android.os.Build.VERSION.RELEASE)
                      .append(" | SDK ").append(android.os.Build.VERSION.SDK_INT);
                String[] labels={"subgroup-reduction","output-major","fused-4"};
                // A 3x3 Latin-square schedule places each candidate in every
                // run position once, reducing systematic first/last-order bias.
                int[][] schedule={{0,1,2},{1,2,0},{2,0,1}};
                double[][] samples=new double[3][3];
                int[] counts=new int[3];
                java.util.regex.Pattern timing=java.util.regex.Pattern.compile("median GPU ([0-9.]+) ms");
                for(int round=0;round<schedule.length&&!stopRequested;round++){
                    for(int slot=0;slot<3&&!stopRequested;slot++){
                        int which=schedule[round][slot];
                        String result=which==0?VulkanProbe.subgroup():
                            which==1?VulkanProbe.gemv(1):VulkanProbe.fused(4);
                        report.append("\nRound ").append(round+1).append(" position ").append(slot+1)
                              .append(" [").append(labels[which]).append("]: ").append(result);
                        java.util.regex.Matcher match=timing.matcher(result);
                        if(match.find()&&counts[which]<3)
                            samples[which][counts[which]++]=Double.parseDouble(match.group(1));
                    }
                }
                report.append("\nBalanced comparison summary (median of up to 3 independent runs):");
                for(int i=0;i<labels.length;i++){
                    if(counts[i]==3){
                        java.util.Arrays.sort(samples[i]);
                        double median=samples[i][1],spread=samples[i][2]-samples[i][0];
                        report.append("\n").append(labels[i]).append(": ").append(String.format(java.util.Locale.US,
                            "median %.3f ms; range %.3f–%.3f ms; range/median %.1f%%",
                            median,samples[i][0],samples[i][2],100.0*spread/median));
                    }else{
                        report.append("\n").append(labels[i]).append(": incomplete (")
                              .append(counts[i]).append("/3 valid runs)");
                    }
                }
                report.append("\nMethod: 3 Latin-square rounds; every kernel appears once in each position. ")
                      .append("Each run uses 32 warm-up and 11 measured submissions. GPU timestamps exclude setup. ")
                      .append("Synthetic weights only; this does not measure model latency or RTF.");
            }catch(Exception|OutOfMemoryError e){report.append("\nComparison failed: ").append(e);}
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
