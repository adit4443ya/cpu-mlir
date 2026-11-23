#!/usr/bin/env bash
# build_vl_probe.sh [OUT]: build vl_probe.c as a static AArch64 exe, same flags as the driver's launcher.
# Needs a sourced cpu-mlir env (LLVM_BINARY_DIR, ARM64_TOOLCHAIN_DIR).
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd); TC=$ARM64_TOOLCHAIN_DIR
$LLVM_BINARY_DIR/clang -O2 --target=aarch64-linux-gnu -march=armv8.5a+sve2+sme \
  --sysroot=$TC/aarch64-none-linux-gnu/libc --gcc-toolchain=$TC -static $D/vl_probe.c -o "${1:-vl_probe}"
echo "built ${1:-vl_probe}"
