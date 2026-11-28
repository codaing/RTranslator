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
adb logcat -d -s RealProjectionProbe:I '*:S' > install-diagnostics/real-projection-probe.txt
grep -F 'FP32_GPU_ONLY_PASS' install-diagnostics/webgpu-probe.txt
grep -E 'FP16_CHAIN_(PASS|NOT_TESTED)' install-diagnostics/webgpu-probe.txt
grep -F 'Vulkan real-model fused-4 projection' install-diagnostics/real-projection-probe.txt | grep -F ': PASS'
grep -E 'OK \(10 tests\)' install-diagnostics/instrumentation.txt
