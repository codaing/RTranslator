package nie.translator.seamless;

import android.app.Activity;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Intent;
import android.database.Cursor;
import android.net.Uri;
import android.os.Bundle;
import android.provider.OpenableColumns;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.Spinner;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class MainActivity extends Activity {
    private static final int REQUEST_MODEL_ZIP = 1;
    private static final int REQUEST_WAV = 2;

    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private TextView modelInfo;
    private TextView wavInfo;
    private TextView status;
    private TextView reportView;
    private Button importModels;
    private Button chooseWav;
    private Button runBenchmark;
    private Button cancelBenchmark;
    private Button copyReport;
    private Spinner schedule;
    private CheckBox quietCpu;

    private Uri selectedWav;
    private String selectedWavName;
    private volatile boolean busy;
    private volatile boolean benchmarkRunning;
    private volatile boolean cancelRequested;
    private volatile boolean closed;
    private volatile String lastReport;

    private File modelDir() {
        return new File(getFilesDir(), "models");
    }

    @Override
    public void onCreate(Bundle state) {
        super.onCreate(state);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        if (state != null) {
            String uri = state.getString("selectedWav");
            if (uri != null) selectedWav = Uri.parse(uri);
            selectedWavName = state.getString("selectedWavName");
            lastReport = state.getString("lastReport");
        }

        LinearLayout panel = new LinearLayout(this);
        panel.setOrientation(LinearLayout.VERTICAL);
        panel.setPadding(36, 44, 36, 36);

        TextView title = new TextView(this);
        title.setText("WAV RTF Benchmark");
        title.setTextSize(26);
        panel.addView(title);

        TextView intro = new TextView(this);
        intro.setText("Standalone offline benchmark for the experimental Seamless model. No microphone, playback, network or RTranslator integration. Use the same WAV and model pack when comparing runs.");
        intro.setTextSize(15);
        panel.addView(intro);

        TextView modelTitle = new TextView(this);
        modelTitle.setText("1. Model package");
        modelTitle.setTextSize(18);
        panel.addView(modelTitle);

        modelInfo = new TextView(this);
        panel.addView(modelInfo);

        importModels = new Button(this);
        importModels.setText("Import model package (.zip)");
        panel.addView(importModels);
        importModels.setOnClickListener(v -> openModelPicker());

        TextView wavTitle = new TextView(this);
        wavTitle.setText("2. Audio fixture");
        wavTitle.setTextSize(18);
        panel.addView(wavTitle);

        TextView wavHelp = new TextView(this);
        wavHelp.setText("WAV format required: uncompressed PCM, mono, 16 kHz, signed 16-bit; 1–30 seconds. No automatic resampling.");
        panel.addView(wavHelp);

        wavInfo = new TextView(this);
        panel.addView(wavInfo);

        chooseWav = new Button(this);
        chooseWav.setText("Choose WAV file");
        panel.addView(chooseWav);
        chooseWav.setOnClickListener(v -> openWavPicker());

        TextView benchmarkTitle = new TextView(this);
        benchmarkTitle.setText("3. Run benchmark");
        benchmarkTitle.setTextSize(18);
        panel.addView(benchmarkTitle);

        TextView benchmarkHelp = new TextView(this);
        benchmarkHelp.setText("CPU baseline · 4 intra-op threads · 1 inter-op thread. Model loading and session closing are reported separately from processing RTF.");
        panel.addView(benchmarkHelp);

        schedule = new Spinner(this);
        ArrayAdapter<String> profiles = new ArrayAdapter<>(this, android.R.layout.simple_spinner_item,
                new String[]{"Baseline: 0.64 s updates", "Fast: 2.56 s updates", "One pass: EOF only (not streaming)"});
        profiles.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        schedule.setAdapter(profiles);
        schedule.setSelection(0);
        panel.addView(schedule);
        quietCpu = new CheckBox(this);
        quietCpu.setText("Disable idle CPU thread spinning");
        quietCpu.setChecked(false);
        panel.addView(quietCpu);

        runBenchmark = new Button(this);
        runBenchmark.setText("Run CPU RTF benchmark");
        panel.addView(runBenchmark);
        runBenchmark.setOnClickListener(v -> runBenchmark());

        cancelBenchmark = new Button(this);
        cancelBenchmark.setText("Cancel benchmark");
        panel.addView(cancelBenchmark);
        cancelBenchmark.setOnClickListener(v -> cancelBenchmark());

        status = new TextView(this);
        status.setTextSize(15);
        panel.addView(status);

        TextView reportTitle = new TextView(this);
        reportTitle.setText("Benchmark report");
        reportTitle.setTextSize(18);
        panel.addView(reportTitle);

        reportView = new TextView(this);
        reportView.setTextIsSelectable(true);
        reportView.setTextSize(13);
        panel.addView(reportView);

        copyReport = new Button(this);
        copyReport.setText("Copy report");
        panel.addView(copyReport);
        copyReport.setOnClickListener(v -> copyReport());

        ScrollView scroll = new ScrollView(this);
        scroll.addView(panel);
        setContentView(scroll);

        status.setText("Ready. Import the model package once, then choose a WAV file.");
        reportView.setText(lastReport == null ? "No benchmark run yet." : lastReport);
        refreshUi();
    }

    private void openModelPicker() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.setType("*/*");
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        startActivityForResult(intent, REQUEST_MODEL_ZIP);
    }

    private void openWavPicker() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.setType("audio/*");
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        startActivityForResult(intent, REQUEST_WAV);
    }

    @Override
    protected void onActivityResult(int request, int result, Intent data) {
        super.onActivityResult(request, result, data);
        if (result != RESULT_OK || data == null || data.getData() == null || busy) return;

        Uri uri = data.getData();
        try {
            getContentResolver().takePersistableUriPermission(
                    uri, data.getFlags() & Intent.FLAG_GRANT_READ_URI_PERMISSION);
        } catch (SecurityException ignored) {
            // The selected document remains readable for this activity session.
        }

        if (request == REQUEST_WAV) {
            selectedWav = uri;
            selectedWavName = displayName(uri);
            status.setText("WAV selected. Ready to benchmark.");
            refreshUi();
        } else if (request == REQUEST_MODEL_ZIP) {
            importModel(uri);
        }
    }

    private String displayName(Uri uri) {
        try (Cursor cursor = getContentResolver().query(
                uri, new String[]{OpenableColumns.DISPLAY_NAME}, null, null, null)) {
            if (cursor != null && cursor.moveToFirst()) {
                int index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME);
                if (index >= 0) return cursor.getString(index);
            }
        } catch (Exception ignored) {
        }
        return uri.getLastPathSegment() == null ? "Selected audio file" : uri.getLastPathSegment();
    }

    private void importModel(Uri uri) {
        busy = true;
        refreshUi();
        status.setText("Importing and verifying model package. This may take several minutes…");
        worker.execute(() -> {
            String message;
            try (InputStream in = getContentResolver().openInputStream(uri)) {
                if (in == null) throw new IOException("Cannot open selected ZIP");
                ModelPack.importZip(in, modelDir());
                message = "Model package imported and SHA-256 verified.";
            } catch (Exception | OutOfMemoryError e) {
                message = "Model import failed: " + e;
            }
            final String result = message;
            runOnUiThread(() -> {
                busy = false;
                status.setText(result);
                refreshUi();
            });
        });
    }

    private void runBenchmark() {
        if (busy || selectedWav == null || !ModelPack.present(modelDir())) return;
        final Uri wav = selectedWav;
        final String wavName = selectedWavName == null ? displayName(wav) : selectedWavName;
        final int updateFrames = new int[]{64,256,0}[schedule.getSelectedItemPosition()];
        final boolean quiet = quietCpu.isChecked();

        busy = true;
        benchmarkRunning = true;
        cancelRequested = false;
        lastReport = null;
        reportView.setText("Benchmark running…");
        status.setText("Reading WAV and preparing CPU model sessions…");
        refreshUi();

        worker.execute(() -> {
            String result;
            try (InputStream in = getContentResolver().openInputStream(wav)) {
                if (in == null) throw new IOException("Cannot open selected WAV");
                result = OfflineBenchmark.run(
                        modelDir(), in,
                        () -> cancelRequested || closed,
                        this::publishStatus,
                        false, false, quiet, GpuTuning.CPU,
                        wavName, updateFrames);
            } catch (Exception | OutOfMemoryError e) {
                result = "Benchmark failed or was cancelled:\n" + e;
            }
            lastReport = result;
            final String finalResult = result;
            runOnUiThread(() -> {
                reportView.setText(finalResult);
                status.setText(finalResult.startsWith("Benchmark failed") ? "Benchmark did not complete." : "Benchmark complete.");
                busy = false;
                benchmarkRunning = false;
                refreshUi();
            });
        });
    }

    private void publishStatus(String message) {
        runOnUiThread(() -> {
            if (!closed && benchmarkRunning) status.setText(message);
        });
    }

    private void cancelBenchmark() {
        if (!benchmarkRunning) return;
        cancelRequested = true;
        status.setText("Cancellation requested. The current inference step will finish first…");
        refreshUi();
    }

    private void copyReport() {
        if (lastReport == null) return;
        ClipboardManager clipboard = getSystemService(ClipboardManager.class);
        if (clipboard != null) {
            clipboard.setPrimaryClip(ClipData.newPlainText("WAV RTF benchmark", lastReport));
            status.setText("Benchmark report copied to clipboard.");
        }
    }

    private long directoryBytes(File directory) {
        long total = 0;
        File[] children = directory.listFiles();
        if (children == null) return 0;
        for (File child : children) {
            if (child.isDirectory()) total += directoryBytes(child);
            else total += child.length();
        }
        return total;
    }

    private String formatMiB(long bytes) {
        return String.format(java.util.Locale.US, "%.0f MiB", bytes / (1024.0 * 1024.0));
    }

    private void refreshUi() {
        boolean modelReady = ModelPack.present(modelDir());
        modelInfo.setText(modelReady
                ? "Model package ready · " + formatMiB(directoryBytes(modelDir()))
                : "No model package imported. Select the verified model ZIP first.");
        wavInfo.setText(selectedWav == null
                ? "No WAV selected."
                : "Selected: " + selectedWavName);
        schedule.setEnabled(!busy);
        quietCpu.setEnabled(!busy);
        importModels.setEnabled(!busy);
        chooseWav.setEnabled(!busy);
        runBenchmark.setEnabled(!busy && modelReady && selectedWav != null);
        cancelBenchmark.setEnabled(benchmarkRunning && !cancelRequested);
        copyReport.setEnabled(!busy && lastReport != null);
    }

    @Override
    protected void onSaveInstanceState(Bundle state) {
        if (selectedWav != null) state.putString("selectedWav", selectedWav.toString());
        state.putString("selectedWavName", selectedWavName);
        state.putString("lastReport", lastReport);
        super.onSaveInstanceState(state);
    }

    @Override
    protected void onDestroy() {
        closed = true;
        cancelRequested = true;
        worker.shutdown();
        super.onDestroy();
    }
}
