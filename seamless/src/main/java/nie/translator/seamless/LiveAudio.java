package nie.translator.seamless;

import android.media.*;
import java.util.Arrays;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicReference;

/** Concurrent capture/inference/playback. Queue overflow is an explicit failure, never silent data loss. */
final class LiveAudio implements AutoCloseable {
    interface Status {void report(String message);}
    private final AudioManager manager;
    private final Status status;
    private final BlockingQueue<short[]> input=new ArrayBlockingQueue<>(24);
    private final BlockingQueue<float[]> output=new ArrayBlockingQueue<>(8);
    private final AtomicReference<Throwable> error=new AtomicReference<>();
    private volatile boolean running=true, captureDone=false, inferenceDone=false;
    private volatile AudioRecord record;
    private volatile AudioTrack track;
    LiveAudio(AudioManager manager,Status status){this.manager=manager;this.status=status;}
    private static boolean headphones(AudioDeviceInfo d){
        if(d==null)return false;
        int t=d.getType();
        return t==AudioDeviceInfo.TYPE_WIRED_HEADPHONES||t==AudioDeviceInfo.TYPE_WIRED_HEADSET||
            t==AudioDeviceInfo.TYPE_BLUETOOTH_A2DP||t==AudioDeviceInfo.TYPE_BLUETOOTH_SCO||
            t==AudioDeviceInfo.TYPE_USB_HEADSET||t==AudioDeviceInfo.TYPE_BLE_HEADSET;
    }
    private void fail(Throwable t){error.compareAndSet(null,t);running=false;status.report("Stopped: "+t.getMessage());}
    void run(StreamingEngine engine)throws Exception{
        AudioDeviceInfo device=null;
        for(AudioDeviceInfo d:manager.getDevices(AudioManager.GET_DEVICES_OUTPUTS))if(headphones(d)){device=d;break;}
        if(device==null)throw new IllegalStateException("Connect headphones before starting");
        final AudioDeviceInfo chosen=device;
        Thread capture=new Thread(()->{
            try{
                int size=AudioRecord.getMinBufferSize(16000,AudioFormat.CHANNEL_IN_MONO,AudioFormat.ENCODING_PCM_16BIT);
                record=new AudioRecord(MediaRecorder.AudioSource.MIC,16000,AudioFormat.CHANNEL_IN_MONO,AudioFormat.ENCODING_PCM_16BIT,Math.max(size,20480));
                if(record.getState()!=AudioRecord.STATE_INITIALIZED)throw new IllegalStateException("microphone initialization failed");
                record.startRecording();
                while(running){
                    short[] pcm=new short[5120];int n=record.read(pcm,0,pcm.length,AudioRecord.READ_BLOCKING);
                    if(!running)break;
                    if(n<=0)throw new IllegalStateException("microphone read "+n);
                    if(!input.offer(n==pcm.length?pcm:Arrays.copyOf(pcm,n)))throw new IllegalStateException("inference cannot keep up; microphone backlog full");
                }
            }catch(Throwable t){if(running)fail(t);}
            finally{captureDone=true;}
        },"seamless-capture");
        Thread play=new Thread(()->{
            try{
                int min=AudioTrack.getMinBufferSize(16000,AudioFormat.CHANNEL_OUT_MONO,AudioFormat.ENCODING_PCM_FLOAT);
                track=new AudioTrack.Builder().setAudioAttributes(new AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA).setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
                    .setAudioFormat(new AudioFormat.Builder().setSampleRate(16000).setChannelMask(AudioFormat.CHANNEL_OUT_MONO).setEncoding(AudioFormat.ENCODING_PCM_FLOAT).build())
                    .setBufferSizeInBytes(Math.max(min,16000*4)).setTransferMode(AudioTrack.MODE_STREAM).build();
                if(track.getState()!=AudioTrack.STATE_INITIALIZED||!track.setPreferredDevice(chosen))throw new IllegalStateException("headphone route unavailable");
                track.play();
                while(error.get()==null&&(!inferenceDone||!output.isEmpty())){
                    float[] pcm=output.poll(100,TimeUnit.MILLISECONDS);if(pcm==null)continue;
                    if(!headphones(track.getRoutedDevice()))throw new IllegalStateException("headphone route lost");
                    int offset=0;
                    while(offset<pcm.length&&error.get()==null){
                        if(!headphones(track.getRoutedDevice()))throw new IllegalStateException("headphone route lost");
                        int n=track.write(pcm,offset,Math.min(1600,pcm.length-offset),AudioTrack.WRITE_BLOCKING);
                        if(n<=0)throw new IllegalStateException("playback write "+n);
                        offset+=n;
                    }
                }
                // Drain queued audio on a normal stop; stop() does not cut streaming data.
                if(error.get()==null)track.stop();
            }catch(Throwable t){fail(t);}
        },"seamless-playback");
        capture.start();play.start();status.report("Listening — experimental Mandarin to English");
        try{
            StreamingEngine.Sink sink=pcm->{if(!output.offer(pcm,2,TimeUnit.SECONDS))throw new IllegalStateException("audio playback backlog full");};
            while(error.get()==null&&(!captureDone||!input.isEmpty())){
                short[] pcm=input.poll(100,TimeUnit.MILLISECONDS);if(pcm!=null)engine.push(pcm,false,sink);
            }
            if(error.get()==null)engine.push(new short[0],true,sink);
        }catch(Throwable t){fail(t);}
        finally{
            running=false;inferenceDone=true;
            AudioRecord rec=record;if(rec!=null)try{rec.stop();}catch(Exception ignored){}
            capture.join();play.join();
            if(rec!=null)rec.release();
            AudioTrack tr=track;if(tr!=null)tr.release();
        }
        if(error.get()!=null)throw new IllegalStateException("stream failed",error.get());
    }
    @Override public void close(){
        running=false;
        AudioRecord rec=record;if(rec!=null)try{rec.stop();}catch(Exception ignored){}
    }
}
