#!/usr/bin/env bash
# ab_reduction.sh OUTDIR [triton|torch] [ROWS COLS] [EXT]: softmax (one long sum loop per row) in
# the PR #14 variants; per variant: correctness, the reduction loop under llvm-mca, qemu insn count.
#   main        PR #14 pass off
#   pr14        PR #14 as posted (4 accumulators; reassoc set in MLIR but dropped for masked reductions)
#   pr14fix     PR #14 + masked reassoc reductions rewritten to select + unmasked reduction
#   reassocfix  reassoc + that rewrite, no unrolling (1 accumulator)
# EXT=sme also takes SSVE=1 (outline reductions into streaming functions).
set -euo pipefail
T=$(cd "$(dirname "$0")" && pwd); OUT=$(realpath -m "$1"); FE=${2:-triton}; R=${3:-4}; C=${4:-8192}; EXT=${5:-sve}
source ${ENV:-$T/../../env-branch.sh} >/dev/null
export TRITON_ARM64_SAVE_TEMPS=1 TRITON_ALWAYS_COMPILE=1 QEMU_CPU=${QEMU_CPU:-max,sme_fa64=off}
[ "${SSVE:-0}" = 1 ] && export TRITON_ARM64_SSVE_REDUCTIONS=1
for v in ${VARIANTS:-main pr14 pr14fix reassocfix}; do
  case $v in main) e="TRITON_ARM64_NO_REDUCTION_SCHEDULING=1";; pr14) e="";;
    pr14fix) e="TRITON_ARM64_REDSCHED_UNMASK=1";; reassocfix) e="TRITON_ARM64_REDSCHED_REASSOC_ONLY=1 TRITON_ARM64_REDSCHED_UNMASK=1";; esac
  rm -rf $OUT/$v; mkdir -p $OUT/$v; cd $OUT/$v
  echo "== $v ($FE ${R}x$C $EXT ssve=${SSVE:-0}): $(env $e python $T/../softmax/run_one.py $FE $R $C $EXT 2>&1 | tail -1)"
  s=$(ls *-arm64.S 2>/dev/null | head -1 || true); [ -z "$s" ] && continue; python $T/mca_loop.py $s
  exe=$(ls *.exe | head -1); if [ -n "${QPROF:-}" ]; then python $T/qprof.py $OUT/$v/$exe "$QPROF" | tail -4; fi
done
