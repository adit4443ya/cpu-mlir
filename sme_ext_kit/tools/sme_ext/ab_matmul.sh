#!/usr/bin/env bash
# ab_matmul.sh OUTDIR [M K N]: Triton matmul (test_matmul.py kernel) on SME in 5 variants,
# each compiled in OUTDIR/<variant> with all temps, then profiled with qprof.py.
#   base    = main behaviour (no accumulator-pack hoist, loops with results not outlined)
#   hoist   = accumulator pack/unpack hoisted out of the k-loop
#   outline = k-loop outlined into <kernel>_sme_0 (kernel itself non-streaming)
#   both    = hoist + outline
#   all     = both + per-iteration buffers allocated once before the loop (BufferLoopHoisting)
set -euo pipefail
T=$(cd "$(dirname "$0")" && pwd); OUT=$(realpath -m "$1"); shift
source ${ENV:-$T/../../env-branch.sh} >/dev/null
export TRITON_ARM64_SAVE_TEMPS=1 TRITON_ALWAYS_COMPILE=1 QEMU_CPU=${QEMU_CPU:-max,sme_fa64=off}
for v in ${VARIANTS:-base hoist outline both all}; do
  mkdir -p $OUT/$v; cd $OUT/$v
  case $v in base) e="TRITON_ARM64_NO_PACK_HOIST=1 TRITON_ARM64_NO_OUTLINE_RESULTS=1 TRITON_ARM64_NO_BUFFER_HOIST=1";;
             hoist) e="TRITON_ARM64_NO_OUTLINE_RESULTS=1 TRITON_ARM64_NO_BUFFER_HOIST=1";; outline) e="TRITON_ARM64_NO_PACK_HOIST=1 TRITON_ARM64_NO_BUFFER_HOIST=1";; both) e="TRITON_ARM64_NO_BUFFER_HOIST=1";; all) e="";; esac
  echo "== $v: $(env $e python $T/run_matmul.py sme "$@" 2>&1 | tail -1)"
  python $T/qprof.py $OUT/$v/matmul_kernel.exe
done
