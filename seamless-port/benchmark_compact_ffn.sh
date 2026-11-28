#!/usr/bin/env bash
# All device file I/O runs under the benchmark app UID.
set -euo pipefail
cd "$(dirname "$0")"
PKG=org.tensorflow.lite.benchmark
REL=files/seamless-compact
PYTHON=${PYTHON:-python3}
app_sh() { adb shell -T "run-as $PKG sh -c '$1'"; }
adb get-state
adb shell -T "run-as $PKG id"
APP_HOME=$(adb shell -T "run-as $PKG pwd" | tr -d '\r\n')
[[ "$APP_HOME" == /data/* ]] || { echo "Unexpected app directory" >&2; exit 1; }
REMOTE="$APP_HOME/$REL"
"$PYTHON" -c 'import numpy'
app_sh "mkdir -p $REL"
for f in first-row.input.bin ffn-m1-compact-k448-fp32.tflite ffn-m1-compact-k512-fp32.tflite ffn-m1-compact-k1024-fp32.tflite; do
    test -s "$f"
    app_sh "cat > $REL/$f" < "$f"
    local_size=$(wc -c < "$f" | tr -d '[:space:]')
    remote_size=$(app_sh "wc -c < $REL/$f" | tr -d '[:space:]')
    [[ "$local_size" == "$remote_size" ]]
done
for k in 448 512 1024; do
    name=ffn-m1-compact-k$k-fp32
    out=$name-cpu-output.bin
    log=$name-cpu.log
    app_sh "rm -f $REL/$out"
    rm -f "$out" "$out.part"
    since=$(adb shell -T "date '+%m-%d %H:%M:%S'" | tr -d '\r\n').000
    args="--graph=$REMOTE/$name.tflite --input_layer=activation --input_layer_shape=1,1,1,1024 --input_layer_value_files=activation:$REMOTE/first-row.input.bin --num_threads=4 --warmup_runs=10 --num_runs=100 --use_nnapi=false --use_xnnpack=true --output_filepath=$REMOTE/$out"
    # ActivityManager starts the Activity under the app UID.
    adb shell -T "am start -S -n $PKG/.BenchmarkModelActivity --es args \"$args\""
    got=0
    for ((i=0;i<180;i++)); do
        got=$(app_sh "if test -f $REL/$out; then wc -c < $REL/$out; else echo 0; fi" | tr -d '[:space:]')
        [[ "$got" == 4096 ]] && break
        sleep 1
    done
    [[ "$got" == 4096 ]] && sleep 20
    adb shell -T "logcat -d -v threadtime -T '$since' -s tflite_BenchmarkModelActivity tflite ExecutionPlan Darwinn" > "$log"
    [[ "$got" == 4096 ]] || { echo "Output missing: $log" >&2; exit 1; }
    adb exec-out run-as "$PKG" cat "$REL/$out" > "$out.part"
    [[ $(wc -c < "$out.part" | tr -d '[:space:]') == 4096 ]]
    mv "$out.part" "$out"
done
"$PYTHON" audit_compact_ffn.py . | tee COMPACT-DEVICE-ACCURACY.json
