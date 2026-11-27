#!/usr/bin/env bash
set -euo pipefail
mkdir -p install-diagnostics
APK=app/build/outputs/apk/release/app-release.apk
{
  echo "=== Emulator OS, ABIs and page size ==="
  adb shell getprop ro.build.version.release
  adb shell getprop ro.build.version.sdk
  ABIS="$(adb shell getprop ro.product.cpu.abilist | tr -d '\r')"
  echo "ABI list: $ABIS"
  adb shell getconf PAGE_SIZE
  adb shell df -h /data
  if [[ ",$ABIS," != *,arm64-v8a,* ]]; then
    echo "INSTALL_TEST=SKIPPED (emulator does not support ARM64; cannot test this APK)"
    exit 0
  fi
  echo "=== Clean install of exact release APK ==="
  adb logcat -c
  set +e
  adb install --abi arm64-v8a --no-streaming "$APK"
  status=$?
  set -e
  echo "adb install exit code: $status"
  adb logcat -d -t 3000 > install-diagnostics/emulator-logcat.txt || true
  if [[ "$status" -ne 0 ]]; then
    echo "INSTALL_TEST=FAIL"
    grep -Ei 'PackageManager|PackageInstaller|installd|INSTALL_FAILED|NativeLibraryHelper' install-diagnostics/emulator-logcat.txt | tail -n 80 || true
    exit "$status"
  fi
  adb shell pm path nie.translator.rtranslator
  echo "INSTALL_TEST=PASS"
} 2>&1 | tee install-diagnostics/emulator-install.txt
