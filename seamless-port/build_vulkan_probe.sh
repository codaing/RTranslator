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
for subgroup in 16 32; do
  "$GLSLC" --target-env=vulkan1.1 -O -DSG_SIZE=$subgroup seamless-port/vulkan_probe_subgroup.comp -o "$TMP/gemv_subgroup$subgroup.spv"
done
for outputs in 4 8 16; do
  "$GLSLC" --target-env=vulkan1.0 -O -DOUTS=$outputs seamless-port/vulkan_probe_fused.comp -o "$TMP/gemv_fused$outputs.spv"
done
"$GLSLC" --target-env=vulkan1.0 -O seamless-port/vulkan_real_projection.comp -o "$TMP/real_projection.spv"
for rows in 2 4 8; do
  "$GLSLC" --target-env=vulkan1.0 -O -DROWS=$rows seamless-port/vulkan_real_projection_multirow.comp -o "$TMP/real_projection_rows$rows.spv"
  rows_per_subgroup=$((rows / 2))
  "$GLSLC" --target-env=vulkan1.1 -O -DROWS_PER_SUBGROUP=$rows_per_subgroup seamless-port/vulkan_real_projection_subgroup16.comp -o "$TMP/real_projection_subgroup16_rows$rows.spv"
done
python - "$TMP" <<'PY'
import pathlib,struct,sys
d=pathlib.Path(sys.argv[1])
for name in ("probe","matmul","gemv","gemv_subgroup16","gemv_subgroup32","gemv_fused4","gemv_fused8","gemv_fused16","real_projection","real_projection_rows2","real_projection_rows4","real_projection_rows8","real_projection_subgroup16_rows2","real_projection_subgroup16_rows4","real_projection_subgroup16_rows8"):
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
