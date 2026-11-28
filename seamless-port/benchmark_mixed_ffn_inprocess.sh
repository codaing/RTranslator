#!/usr/bin/env bash
# In-process mixed FFN benchmark. All app-private file operations use run-as.
set -euo pipefail
cd "$(dirname "$0")"
PKG=nie.translator.seamless
TEST_PKG=nie.translator.seamless.test
REL=files/seamless-mixed
app_sh() { adb shell -T "run-as $PKG sh -c '$1'"; }

adb get-state
adb install -r seamless-debug.apk
adb install -r seamless-debug-androidTest.apk
adb shell -T "run-as $PKG id"
app_sh "mkdir -p $REL"

for f in ffn-m1-compact-k512-fp32.tflite \
    ffn-m31-m31-prefill-tail-g64-int8.tflite \
    tail-manifest.json mixed-full32-input.f32 mixed-full32-reference.f32; do
    test -s "$f"
    app_sh "cat > $REL/$f" < "$f"
    local_size=$(wc -c < "$f" | tr -d '[:space:]')
    remote_size=$(app_sh "wc -c < $REL/$f" | tr -d '[:space:]')
    [[ "$local_size" == "$remote_size" ]] || { echo "Transfer mismatch: $f" >&2; exit 1; }
done

adb logcat -c
adb shell -T am instrument -w -r -e class nie.translator.seamless.MixedFfnRuntimeTest \
    "$TEST_PKG/androidx.test.runner.AndroidJUnitRunner" | tee MIXED-INPROCESS-INSTRUMENTATION.txt
adb logcat -d -v threadtime -s MixedFfnRuntime tflite ExecutionPlan > MIXED-INPROCESS-DEVICE.log
grep -F "compilation finished successfully on google-edgetpu" MIXED-INPROCESS-DEVICE.log
grep -F "Replacing 17 out of 17 node(s) with delegate (TfLiteNnapiDelegate)" MIXED-INPROCESS-DEVICE.log
grep -F "Replacing 2 out of 2 node(s) with delegate (TfLiteXNNPackDelegate)" MIXED-INPROCESS-DEVICE.log
grep -F "MIXED_FFN_INPROCESS_PASS" MIXED-INPROCESS-DEVICE.log
adb exec-out run-as "$PKG" cat "$REL/MIXED-INPROCESS-RESULT.json" > MIXED-INPROCESS-RESULT.json
test -s MIXED-INPROCESS-RESULT.json
"$PYTHON" -m json.tool MIXED-INPROCESS-RESULT.json
