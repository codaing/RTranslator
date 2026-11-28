#!/usr/bin/env bash
set -euo pipefail
NDK="$ANDROID_HOME/ndk/28.2.13676358"
BIN="$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin"
GLSLC="$NDK/shader-tools/linux-x86_64/glslc"
test -x "$GLSLC"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
"$GLSLC" --target-env=vulkan1.0 -O seamless-port/vulkan_probe.comp -o "$TMP/probe.spv"
"$GLSLC" --target-env=vulkan1.0 -O seamless-port/vulkan_probe_matmul.comp -o "$TMP/matmul.spv"
"$GLSLC" --target-env=vulkan1.0 -O seamless-port/vulkan_probe_gemv.comp -o "$TMP/gemv.spv"
for outputs in 4 8 16; do
  "$GLSLC" --target-env=vulkan1.0 -O -DOUTS=$outputs seamless-port/vulkan_probe_fused.comp -o "$TMP/gemv_fused$outputs.spv"
done
python - "$TMP" <<'PY'
import pathlib,struct,sys
d=pathlib.Path(sys.argv[1])
for name in ("probe","matmul","gemv","gemv_fused4","gemv_fused8","gemv_fused16"):
    data=(d/(name+".spv")).read_bytes()
    words=struct.unpack("<"+"I"*(len(data)//4),data)
    assert words[0]==0x07230203
    (d/(name+"_shader.h")).write_text("#include <stdint.h>\nstatic const uint32_t "+name+"_shader[]={"+",".join(hex(x) for x in words)+"};\n")
PY
OUT=seamless/src/main/jniLibs/arm64-v8a
mkdir -p "$OUT"
"$BIN/aarch64-linux-android28-clang++" -std=c++17 -O2 -fPIC -shared -static-libstdc++ \
  -Wl,-z,max-page-size=16384 -Wl,--no-undefined -I"$TMP" \
  seamless-port/vulkan_probe.cpp -lvulkan -o "$OUT/libseamless_vulkan_probe.so"
