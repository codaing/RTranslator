#!/usr/bin/env bash
set -euo pipefail
mkdir -p install-diagnostics
adb shell getprop ro.product.cpu.abilist | tee install-diagnostics/emulator.txt
adb shell getconf PAGE_SIZE | tee -a install-diagnostics/emulator.txt
adb install --abi arm64-v8a --no-streaming seamless/build/outputs/apk/release/seamless-release.apk
adb shell am start -W -n nie.translator.seamless/.MainActivity
adb shell pidof nie.translator.seamless
adb uninstall nie.translator.seamless
adb install --abi arm64-v8a --no-streaming seamless/build/outputs/apk/debug/seamless-debug.apk
adb install --no-streaming seamless/build/outputs/apk/androidTest/debug/seamless-debug-androidTest.apk
adb shell am instrument -w -r nie.translator.seamless.test/androidx.test.runner.AndroidJUnitRunner | tee install-diagnostics/instrumentation.txt
adb logcat -d -s LowBitProbe:I '*:S' > install-diagnostics/lowbit-probe.txt
adb logcat -d -s WebGpuProbe:I '*:S' > install-diagnostics/webgpu-probe.txt
grep -F 'FP32_GPU_ONLY_PASS' install-diagnostics/webgpu-probe.txt
grep -E 'FP16_CHAIN_(PASS|NOT_TESTED)' install-diagnostics/webgpu-probe.txt
if [[ "${REQUIRE_REAL_PROJECTION:-0}" == "1" ]]; then
  adb logcat -d -s RealProjectionProbe:I '*:S' > install-diagnostics/real-projection-probe.txt
  for variant in fused-4 multirow-2 multirow-4 multirow-8; do
    grep -F "Vulkan real-model $variant projection" install-diagnostics/real-projection-probe.txt | grep -F ': PASS'
  done
fi
grep -E 'OK \((9|10) tests\)' install-diagnostics/instrumentation.txt
