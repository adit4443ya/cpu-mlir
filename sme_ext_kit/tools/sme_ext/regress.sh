#!/usr/bin/env bash
# regress.sh OUTDIR: the whole test suite, sequential: backend lit, then the Triton and torch pytest
# suites on each target. Writes OUTDIR/summary.txt plus one log per suite.
#   ENV=<env file>    default: ../../env-branch.sh (laptop); on dev compute: $WORK/cpu-mlir-env.sh
#   TARGETS="sme sve" default; RUN_ON_SIM=0 + DEV_HOST/ADB_PATH/DEV_ID/TRITON_ARM64_STAGING_DIR = run on the phone
#   QEMU_CPU          default max,sme_fa64=off (qemu's plain `max` has FA64, which hides streaming bugs)
OUT=$(realpath -m "$1"); mkdir -p $OUT
source ${ENV:-$(dirname "$0")/../../env-branch.sh} >/dev/null
export QEMU_CPU=${QEMU_CPU:-max,sme_fa64=off}
P=$ARM64_BACKEND_PATH/python; S=$OUT/summary.txt; : > $S
lit $TRITON_BUILD_DIR/third_party/qcom_arm64_backend/test > $OUT/lit.log 2>&1
echo "lit: $(grep -E "^ +(Passed|Failed|Unresolved|Unsupported|Expectedly)" $OUT/lit.log | tr -s ' ' | paste -sd' ')" | tee -a $S
for ext in ${TARGETS:-sme sve}; do
  (cd $P/examples && python -m pytest -q --target $ext . -p no:cacheprovider > $OUT/triton_$ext.log 2>&1); echo "triton $ext: $(tail -1 $OUT/triton_$ext.log)" | tee -a $S
  (cd $P/torch-mlir && python -m pytest -q --target $ext . -p no:cacheprovider > $OUT/torch_$ext.log 2>&1); echo "torch  $ext: $(tail -1 $OUT/torch_$ext.log)" | tee -a $S
done
