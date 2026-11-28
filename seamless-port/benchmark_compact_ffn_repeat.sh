#!/usr/bin/env bash
# Six-round balanced compact-FFN repeatability test.
# Every on-device file operation is performed as the benchmark app UID.
set -euo pipefail
cd "$(dirname "$0")"
PKG=org.tensorflow.lite.benchmark
REL=files/seamless-compact
PYTHON=\${PYTHON:-python3}
ROUNDS=\${ROUNDS:-6}
NUM_RUNS=\${NUM_RUNS:-1000}
WARMUP_RUNS=\${WARMUP_RUNS:-50}
REST_SECONDS=\${REST_SECONDS:-8}
TIMEOUT_SECONDS=\${TIMEOUT_SECONDS:-180}
app_sh() { adb shell -T "run-as $PKG sh -c '$1'"; }
adb get-state
adb shell -T "run-as $PKG id"
APP_HOME=$(adb shell -T "run-as $PKG pwd" | tr -d '\r\n')
[[ "$APP_HOME" == /data/* ]] || { echo "Unexpected app-private directory: $APP_HOME" >&2; exit 1; }
REMOTE="$APP_HOME/$REL"
"$PYTHON" -c 'import numpy'
app_sh "mkdir -p $REL"
for f in first-row.input.bin first-row.source-reference.bin \
    ffn-m1-compact-k448-fp32.tflite ffn-m1-compact-k512-fp32.tflite \
    ffn-m1-compact-k1024-fp32.tflite; do
    test -s "$f"
    app_sh "cat > $REL/$f" < "$f"
    local_size=$(wc -c < "$f" | tr -d '[:space:]')
    remote_size=$(app_sh "wc -c < $REL/$f" | tr -d '[:space:]')
    [[ "$local_size" == "$remote_size" ]] || { echo "Transfer size mismatch: $f" >&2; exit 1; }
done
# Each width appears twice in every ordinal position across these six rounds.
orders=(
    "448 512 1024"
    "512 1024 448"
    "1024 448 512"
    "448 1024 512"
    "1024 512 448"
    "512 448 1024"
)
if (( ROUNDS < 1 || ROUNDS > \${#orders[@]} )); then
    echo "ROUNDS must be between 1 and \${#orders[@]}" >&2
    exit 2
fi
for ((round=1; round<=ROUNDS; round++)); do
    echo "=== compact FFN repeatability round $round/$ROUNDS ==="
    read -r -a order <<< "\${orders[$((round-1))]}"
    for k in "\${order[@]}"; do
        name="ffn-m1-compact-k\${k}-fp32"
        out="repeat-r\${round}-k\${k}.bin"
        log="repeat-r\${round}-k\${k}.log"
        device="repeat-r\${round}-k\${k}-device.txt"
        app_sh "rm -f $REL/$out"
        rm -f "$out" "$out.part" "$log" "$device"
        {
            echo "round=$round width=$k"
            adb shell -T getprop ro.product.model
            adb shell -T getprop ro.build.fingerprint
            adb shell -T dumpsys battery
            adb shell -T dumpsys thermalservice 2>&1 || true
        } > "$device"
        since=$(adb shell -T "date '+%m-%d %H:%M:%S'" | tr -d '\r\n').000
        args="--graph=$REMOTE/$name.tflite --input_layer=activation --input_layer_shape=1,1,1,1024 --input_layer_value_files=activation:$REMOTE/first-row.input.bin --num_threads=4 --warmup_runs=$WARMUP_RUNS --num_runs=$NUM_RUNS --use_nnapi=false --use_xnnpack=true --output_filepath=$REMOTE/$out"
        echo "round=$round width=$k runs=$NUM_RUNS warmup=$WARMUP_RUNS"
        adb shell -T "am start -S -n $PKG/.BenchmarkModelActivity --es args \"$args\""
        got=0
        for ((i=0; i<TIMEOUT_SECONDS; i++)); do
            got=$(app_sh "if test -f $REL/$out; then wc -c < $REL/$out; else echo 0; fi" | tr -d '[:space:]')
            [[ "$got" == 4096 ]] && break
            sleep 1
        done
        if [[ "$got" != 4096 ]]; then
            adb logcat -d -v threadtime -T "$since" -s tflite_BenchmarkModelActivity tflite ExecutionPlan Darwinn > "$log" || true
            echo "Timed out waiting for output: $out" >&2
            exit 1
        fi
        sleep 2
        adb logcat -d -v threadtime -T "$since" -s tflite_BenchmarkModelActivity tflite ExecutionPlan Darwinn > "$log"
        adb exec-out run-as "$PKG" cat "$REL/$out" > "$out.part"
        [[ $(wc -c < "$out.part" | tr -d '[:space:]') == 4096 ]] || { echo "Pulled output size mismatch: $out" >&2; exit 1; }
        mv "$out.part" "$out"
        sleep "$REST_SECONDS"
    done
done
"$PYTHON" summarize_compact_ffn_repeat.py . | tee COMPACT-REPEATABILITY.json
