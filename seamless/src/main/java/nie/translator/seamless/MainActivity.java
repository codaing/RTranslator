package nie.translator.seamless;

import android.app.Activity;
import android.os.Bundle;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Button;
import android.content.Intent;
import android.net.Uri;
import android.media.AudioAttributes;
import android.media.AudioFormat;
import android.media.AudioTrack;
import java.io.File;
import java.io.InputStream;
import java.io.FileOutputStream;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class MainActivity extends Activity {
    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private TextView status;
    private Button test;
    private volatile boolean closed;
    private volatile AudioTrack track;

    @Override public void onCreate(Bundle saved) {
        super.onCreate(saved);
        LinearLayout panel = new LinearLayout(this);
        panel.setOrientation(LinearLayout.VERTICAL);
        panel.setPadding(32, 48, 32, 32);
        TextView title = new TextView(this);
        title.setText("Mandarin to English — SeamlessStreaming port");
        title.setTextSize(24);
        panel.addView(title);
        status = new TextView(this);
        status.setText("Development test only. Live translation is not implemented. Import a self-contained validated vocoder.onnx to test acoustic-unit synthesis; the encoder and unit decoder are not connected.");
        panel.addView(status);
        Button load = new Button(this);
        load.setText("Import vocoder.onnx");
        load.setOnClickListener(v -> {
            Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
            intent.setType("*/*");
            intent.addCategory(Intent.CATEGORY_OPENABLE);
            startActivityForResult(intent, 17);
        });
        panel.addView(load);
        test = new Button(this);
        test.setText("Test vocoder (synthetic units, not translation)");
        test.setEnabled(model().isFile());
        test.setOnClickListener(v -> runVocoder());
        panel.addView(test);
        setContentView(panel);
    }

    private File model() { return new File(getFilesDir(), "vocoder.onnx"); }
    private void show(String message, boolean enable) {
        runOnUiThread(() -> {
            if (!closed) { status.setText(message); test.setEnabled(enable); }
        });
    }
    @Override protected void onActivityResult(int request, int result, Intent data) {
        super.onActivityResult(request, result, data);
        if (request != 17 || result != RESULT_OK || data == null || data.getData() == null) return;
        Uri uri = data.getData();
        test.setEnabled(false);
        worker.execute(() -> {
            File temp = new File(getFilesDir(), "vocoder.tmp");
            try (InputStream in = getContentResolver().openInputStream(uri);
                 FileOutputStream out = new FileOutputStream(temp)) {
                if (in == null) throw new java.io.IOException("Cannot open document");
                byte[] block = new byte[65536];
                int n;
                while ((n = in.read(block)) != -1) out.write(block, 0, n);
                out.getFD().sync();
                if (!temp.renameTo(model())) throw new java.io.IOException("Cannot save model");
                show("Vocoder imported. This does not enable translation.", true);
            } catch (Exception e) {
                temp.delete();
                show("Import failed: " + e.getMessage(), model().isFile());
            }
        });
    }
    private void runVocoder() {
        test.setEnabled(false);
        worker.execute(() -> {
            try (EnglishUnitVocoder vocoder = new EnglishUnitVocoder(model().getAbsolutePath())) {
                long[] units = new long[50];
                for (int i = 0; i < units.length; i++) units[i] = 100 + i;
                float[] pcm = vocoder.decode(units);
                if (pcm.length == 0) throw new IllegalStateException("No samples returned");
                if (closed) return;
                AudioTrack audio = new AudioTrack.Builder()
                    .setAudioAttributes(new AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA).build())
                    .setAudioFormat(new AudioFormat.Builder().setSampleRate(16000)
                        .setEncoding(AudioFormat.ENCODING_PCM_FLOAT).setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build())
                    .setBufferSizeInBytes(pcm.length * 4).setTransferMode(AudioTrack.MODE_STATIC).build();
                track = audio;
                try {
                    int written = audio.write(pcm, 0, pcm.length, AudioTrack.WRITE_BLOCKING);
                    if (written != pcm.length) throw new IllegalStateException("Audio write: " + written);
                    if (!closed) {
                        audio.play();
                        Thread.sleep(Math.min(10000, pcm.length * 1000L / 16000 + 100));
                    }
                } finally { track = null; audio.release(); }
                show("Vocoder returned " + pcm.length + " samples. Synthetic acoustic-unit test only.", true);
            } catch (Exception e) { show("Vocoder failed: " + e.getMessage(), true); }
        });
    }
    @Override protected void onDestroy() {
        closed = true;
        worker.shutdownNow();
        super.onDestroy();
    }
}
