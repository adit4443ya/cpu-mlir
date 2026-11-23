#!/usr/bin/env bash
# device_bench.sh OUTDIR SUITE: time the variants of one suite on the phone (or qemu, for a dry run).
#   SUITE=matmul   needs the build with sme-outline-ext.patch (or the combined patch)
#                  variants: base (= main: all three matmul changes switched off) | all (this branch)
#   SUITE=softmax  needs the build with combined_with_pr14.patch
#                  variants: main (PR #14 off) | pr14 (as posted) | pr14fix | reassocfix, on sve and sme,
#                  plus ssve_reassocfix / ssve_pr14fix on sme (reductions in streaming mode)
# Env: ENV (env file to source), REPS (default 5), ITERS (benchmark iterations inside the exe, default 20),
#      SIZES (override the size list), TARGETS (override), VARIANTS (override), SLEEP (s between runs, default 2).
# Device mode needs RUN_ON_SIM=0 DEV_HOST ADB_PATH DEV_ID TRITON_ARM64_STAGING_DIR set by the caller.
# Every run lands in OUTDIR/raw/<suite>/<target>/<variant>/<size>/rep<k>/ (all temps kept, run.log);
# one line per run is appended to OUTDIR/runs.tsv. Summarize with: python summarize.py OUTDIR
set -uo pipefail
D=$(cd "$(dirname "$0")" && pwd); T=$D/..
OUT=$(realpath -m "$1"); SUITE=$2; mkdir -p $OUT
source ${ENV:?set ENV to your cpu-mlir env file} >/dev/null
export TRITON_ARM64_SAVE_TEMPS=1 TRITON_ALWAYS_COMPILE=1 TRITON_ARM64_BENCHMARK=1
export TRITON_ARM64_BENCHMARK_ITERATIONS=${ITERS:-20} QEMU_CPU=${QEMU_CPU:-max,sme_fa64=off}
MODE=$([ "${RUN_ON_SIM:-1}" = 0 ] && echo device || echo qemu)
[ $MODE = device ] && : ${DEV_HOST:?} ${ADB_PATH:?} ${DEV_ID:?} ${TRITON_ARM64_STAGING_DIR:?}
TSV=$OUT/runs.tsv; [ -f $TSV ] || printf "suite\tmode\ttarget\tvariant\tsize\trep\tms\tmatch\tsmstart_in_asm\tdir\n" > $TSV

envof() {  # variant -> env assignments
  case $1 in
    base) echo "TRITON_ARM64_NO_PACK_HOIST=1 TRITON_ARM64_NO_OUTLINE_RESULTS=1 TRITON_ARM64_NO_BUFFER_HOIST=1";;
    all) echo "";;
    main) echo "TRITON_ARM64_NO_REDUCTION_SCHEDULING=1";;
    pr14) echo "";;
    pr14fix) echo "TRITON_ARM64_REDSCHED_UNMASK=1";;
    reassocfix) echo "TRITON_ARM64_REDSCHED_REASSOC_ONLY=1 TRITON_ARM64_REDSCHED_UNMASK=1";;
    ssve_pr14fix) echo "TRITON_ARM64_SSVE_REDUCTIONS=1 TRITON_ARM64_REDSCHED_UNMASK=1";;
    ssve_reassocfix) echo "TRITON_ARM64_SSVE_REDUCTIONS=1 TRITON_ARM64_REDSCHED_REASSOC_ONLY=1 TRITON_ARM64_REDSCHED_UNMASK=1";;
    *) echo "unknown variant $1" >&2; exit 1;;
  esac
}

run_case() {  # target variant size(words) -> one timed run
  local tgt=$1 var=$2 size=$3 rep=$4 dir cmd ms match sm
  dir=$OUT/raw/$SUITE/$tgt/$var/${size// /x}/rep$rep; rm -rf $dir; mkdir -p $dir
  case $SUITE in
    matmul)  cmd="python $T/run_matmul.py $tgt $size";;
    softmax) cmd="python $T/../softmax/run_one.py $size $tgt";;   # size = "<triton|torch> ROWS COLS"
  esac
  (cd $dir && env $(envof $var) $cmd > run.log 2>&1)
  ms=$(grep -oE "Elapsed time: [0-9.]+" $dir/run.log | tail -1 | awk '{print $3}')
  match=$(grep -oiE "match(ed)?: *(True|False)" $dir/run.log | tail -1 | awk '{print $NF}')
  sm=$(cat $dir/*-arm64.S 2>/dev/null | grep -cE "^\s+smstart" || true)
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" $SUITE $MODE $tgt $var "${size// /x}" $rep "${ms:-NA}" "${match:-NA}" "$sm" "$dir" >> $TSV
  echo "  $tgt $var ${size// /x} rep$rep: ${ms:-NA} ms match=${match:-NA} smstart_in_asm=$sm"
  [ -z "$ms" ] && [ $MODE = device ] && { echo "    no timing, last lines of $dir/run.log:"; tail -5 $dir/run.log | sed 's/^/    /'; }
  sleep ${SLEEP:-2}
}

case $SUITE in
  matmul)  TG=${TARGETS:-sme sve}; VS=${VARIANTS:-base all}
           SZ=${SIZES:-"179 167 321;256 256 256;512 512 512;1024 1024 1024"};;
  softmax) TG=${TARGETS:-sve sme}; VS=${VARIANTS:-main pr14 pr14fix reassocfix ssve_pr14fix ssve_reassocfix}
           SZ=${SIZES:-"triton 4 8192;triton 64 8192;triton 256 2048;torch 64 8192"};;
  *) echo "SUITE must be matmul or softmax"; exit 1;;
esac
[ $MODE = qemu ] && echo "(qemu dry run: the driver times only in device mode, so ms = NA)"
echo "== $SUITE on $MODE: targets [$TG] variants [$VS] reps ${REPS:-5} iters $TRITON_ARM64_BENCHMARK_ITERATIONS"
IFS=';' read -ra SIZELIST <<< "$SZ"
for rep in $(seq 1 ${REPS:-5}); do          # reps outermost: variants interleave, so drift hits all alike
  for size in "${SIZELIST[@]}"; do for tgt in $TG; do for var in $VS; do
    [[ $var == ssve_* && $tgt != sme ]] && continue
    run_case $tgt $var "$size" $rep
  done; done; done
done
