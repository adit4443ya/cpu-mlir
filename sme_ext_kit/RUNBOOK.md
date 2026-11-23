# Runbook for the dev-compute agent: SME outline extension, PR #14, SSVE reductions

You are running on the team's x86 **dev compute** machine, which has the cpu-mlir build and an
Android phone reachable over `ssh $DEV_HOST $ADB_PATH -s $DEV_ID`. This runbook builds two
variants of the compiler, runs the whole test suite (qemu and the phone), times the changes on the
phone, and writes one report. Everything you need is in this kit (`sme_ext_kit/`).

**Getting the kit.** It is on the branch `sme-outline-ext` of `git@github.com:adit4443ya/cpu-mlir.git`,
in the folder `sme_ext_kit/` (the branch's first commit is patch A; the second commit only adds this
kit and is not meant for a PR). Copy the kit out of the checkout before step 2, because step 2 switches
branches and applies patches in `$WORK/cpu-mlir`:
```bash
git clone --depth 2 -b sme-outline-ext git@github.com:adit4443ya/cpu-mlir.git $WORK/sme_ext_src   # no submodules needed
cp -r $WORK/sme_ext_src/sme_ext_kit $WORK/ && export KIT=$WORK/sme_ext_kit
```
Extra reading in the kit: `docs/MATMUL_SME_LOWERING.md` (how the matmul lowers to ZA) and
`docs/SME_OUTLINE_EXT.md` (the full write-up of these changes).

Background, in one paragraph each (details: `docs/SME_OUTLINE_EXT.md` in the author's workspace):
- **Patch A, `patches/sme-outline-ext.patch`** (the one meant for a PR). It does three things for
  the Triton matmul on SME. (1) The C accumulator is packed once before the k-loop and unpacked
  once after it, instead of every iteration. (2) The k-loop is outlined into its own streaming
  function even though it returns values, so the kernel itself is no longer streaming. (3) The
  per-iteration A/B buffers are allocated once (`BufferLoopHoisting`). On qemu: kernel instructions
  41.9M -> 31.9M, and executed `smstart` 2772 -> 144. qemu cannot time mode switches, so the phone
  timing is the missing piece.
- **Patch B, `patches/combined_with_pr14.patch`** = patch A + upstream PR #14 ("add reduction
  scheduling pass": unrolls a scalable float-sum loop 4x with 4 accumulators and marks them
  `reassoc`) + a suggested fix (masked reassoc reductions rewritten as `select` + unmasked
  reduction, because the masked lowering drops `reassoc` and leaves the slow `fadda`) + an
  experimental option that moves reduction loops into streaming-mode functions (**SSVE**).
  llvm-mca predicts the sum loop goes from 12.1 cycles/iteration (main) to 10.0 (PR #14) to
  3.1-3.2 (with the fix, with or without the unroll). SSVE only pays if the phone's streaming
  vector length (SVL) is larger than its SVE vector length.

Both patches carry **temporary env switches**, so one build can produce every variant:

| env var | effect |
|---|---|
| `TRITON_ARM64_NO_PACK_HOIST=1` `TRITON_ARM64_NO_OUTLINE_RESULTS=1` `TRITON_ARM64_NO_BUFFER_HOIST=1` | all three = main's matmul behaviour (variant `base`) |
| `TRITON_ARM64_NO_REDUCTION_SCHEDULING=1` (patch B) | PR #14 off (variant `main`) |
| `TRITON_ARM64_REDSCHED_UNMASK=1` (patch B) | the suggested fix (variant `pr14fix`) |
| `TRITON_ARM64_REDSCHED_REASSOC_ONLY=1` (patch B) | reassoc without the 4x unroll (with UNMASK: variant `reassocfix`) |
| `TRITON_ARM64_SSVE_REDUCTIONS=1` | reductions in streaming mode (variants `ssve_*`, sme only) |

## Rules

- Do **not** commit, push, open PRs or post comments. Do not change anything outside `$WORK`.
- Put every output under `$RES` (defined below). Do not use `/tmp`.
- If a step fails, record the command, the exit code and the last 30 lines of its log in the
  report, then continue with the next step that doesn't depend on it. Never "fix" the compiler
  source yourself.
- Timings are noisy. Do not run anything else heavy on the phone while benchmarking. If the reps
  of one case differ by more than 10 %, rerun that suite with `REPS=9` and report both.
- At the end, put the checkout back on the branch it was on (step 7).

## 0. Variables (fill in, then export in every shell you use)

```bash
export WORK=/local/mnt/workspace/$USER/cpu-mlir-work      # the existing build: $WORK/cpu-mlir, $WORK/cpu-mlir-env.sh
export SETUP=<dir containing build_cpu_mlir_x86.sh>       # the setup kit used to build $WORK
export KIT=<where sme_ext_kit/ was unpacked>/sme_ext_kit
export ENV=$WORK/cpu-mlir-env.sh
export RES=$WORK/sme_ext_results; mkdir -p $RES
export DEV_HOST=<ssh host the phone is plugged into> ADB_PATH=<adb path on that host> DEV_ID=<adb serial>
export TRITON_ARM64_STAGING_DIR=<dir visible at the same path on DEV_HOST>
export QEMU_CPU=max,sme_fa64=off       # qemu's plain `max` has FA64, which hides streaming-mode bugs
source $ENV
```

Rebuild command, used below (incremental, a few minutes; never run bare `ninja`):
```bash
rebuild() { env WORK=$WORK STEP=build JOBS=${JOBS:-4} LINK_JOBS=1 bash $SETUP/build_cpu_mlir_x86.sh > $RES/build_$1.log 2>&1 \
            && echo "build $1 ok" || { echo "build $1 FAILED"; tail -30 $RES/build_$1.log; }; }
```

## 1. Preflight

```bash
C=$WORK/cpu-mlir
git -C $C branch --show-current > $RES/original_branch.txt; git -C $C log --oneline -1 >> $RES/original_branch.txt
git -C $C status --short | grep -v '^??' | tee $RES/status_before.txt
#   expected: only " m triton", " m triton-shared" (setup patches) and maybe the triton_shared_examples link.
#   Any other modified file: STOP and report it; do not discard someone's work.
git -C $C fetch origin && git -C $C log --oneline -1 origin/main | tee $RES/origin_main.txt
#   the patches were made on origin/main = 12579ba; a newer main is fine if they apply (step 2)
ssh $DEV_HOST $ADB_PATH -s $DEV_ID shell echo phone-ok | tee $RES/phone.txt
```

Phone CPU features and vector lengths (decides whether `--target=sme` and SSVE make sense):
```bash
bash $KIT/tools/sme_ext/device/build_vl_probe.sh $RES/vl_probe
ssh $DEV_HOST "mkdir -p $TRITON_ARM64_STAGING_DIR/vl" && scp $RES/vl_probe $DEV_HOST:$TRITON_ARM64_STAGING_DIR/vl/
ssh $DEV_HOST "$ADB_PATH -s $DEV_ID push $TRITON_ARM64_STAGING_DIR/vl/vl_probe /data/local/tmp/ && $ADB_PATH -s $DEV_ID shell 'chmod 755 /data/local/tmp/vl_probe && /data/local/tmp/vl_probe'" | tee $RES/probe.txt
ssh $DEV_HOST $ADB_PATH -s $DEV_ID shell cat /proc/cpuinfo > $RES/cpuinfo.txt
```
If `TRITON_ARM64_STAGING_DIR` is already visible on `DEV_HOST` at the same path (the driver
assumes so), the `scp` is a harmless copy onto itself. Output looks like
`sve=1 sve2=1 sme=1 sme_fa64=0`, `sve_vl_bits=128`, `sme_svl_bits=512`.
**If `sme=0`: skip every `sme` item on the phone** (it would SIGILL) and say so in the report.

## 2. Phase A: patch A (the PR candidate)

```bash
C=$WORK/cpu-mlir
git -C $C checkout -b sme-ext-test-A origin/main
git -C $C apply --whitespace=nowarn $KIT/patches/sme-outline-ext.patch && git -C $C status --short | grep -v '^??'
rebuild A
```
If `git apply` fails on a newer origin/main: `git -C $C checkout -B sme-ext-test-A 12579ba` and apply again
(record that you did).

### 2.1 Whole test suite on qemu (same as the author's laptop run)

```bash
ENV=$ENV RUN_ON_SIM=1 bash $KIT/tools/sme_ext/regress.sh $RES/A_qemu         # ~30 min; lit + Triton + torch on sme and sve
cat $RES/A_qemu/summary.txt
```
Expected (author's laptop, 2026-11-22): lit 15 passed (plus any local scratch files as
Unresolved: list them, don't count them as failures); Triton `195 passed` on sme and on sve;
torch `20 passed` on sme and on sve.

New lit cases fail with the switch that undoes them (proves they test the change). lit does not
pass env vars to the tests, so run the files' RUN lines by hand:
```bash
S=$C/qcom_arm64_backend/test/Transforms; M=$S/arm64-matmul-to-mmt4d-pass.mlir; O=$S/arm64-sme-func-outline.mlir
TRITON_ARM64_NO_PACK_HOIST=1 arm64-mlir-opt $M -split-input-file -triton-arm64-matmul-to-mmt4d 2>&1 | FileCheck $M >/dev/null 2>&1; echo "expect 1: $?"
TRITON_ARM64_NO_OUTLINE_RESULTS=1 arm64-mlir-opt $O -split-input-file -triton-arm64-sme-func-outline 2>&1 | FileCheck $O >/dev/null 2>&1; echo "expect 1: $?"
arm64-mlir-opt $M -split-input-file -triton-arm64-matmul-to-mmt4d | FileCheck $M; echo "expect 0: $?"
arm64-mlir-opt $O -split-input-file -triton-arm64-sme-func-outline | FileCheck $O; echo "expect 0: $?"
```

### 2.2 Whole test suite on the phone

```bash
TARGETS="sve sme" ENV=$ENV RUN_ON_SIM=0 bash $KIT/tools/sme_ext/regress.sh $RES/A_phone   # drop sme if probe says sme=0; slow (adb per test)
cat $RES/A_phone/summary.txt
```
Any failure: put `grep -E "FAILED|Error|SIGILL|signal" $RES/A_phone/*.log | head -40` in the report.
For each failing test also run it on main's behaviour (all three `TRITON_ARM64_NO_*` switches set)
to tell a regression from a pre-existing failure.

### 2.3 Matmul timing on the phone: main behaviour (`base`) vs patch A (`all`)

```bash
ENV=$ENV RUN_ON_SIM=0 REPS=5 ITERS=20 bash $KIT/tools/sme_ext/device/device_bench.sh $RES/A_bench matmul
python $KIT/tools/sme_ext/device/summarize.py $RES/A_bench          # writes $RES/A_bench/RESULTS.md
```
Sizes 179x167x321, 256³, 512³, 1024³ on sme and sve (set `TARGETS=sve` if no SME).
`smstart in asm` should be much lower for `all` than for `base` on sme; `all match` must be `yes`.

### 2.4 Optional: qemu instruction counts (reproduces the author's table)

```bash
ENV=$ENV bash $KIT/tools/sme_ext/ab_matmul.sh $RES/A_qprof     # 5 variants; each prints match + per-symbol instructions + smstart
```
Expected: kernel instructions base 41.92M, hoist 31.87M, all 31.88M; smstart base 2772, all 36 + 108.

## 3. Phase B: patch B (PR #14 + fix + SSVE experiment)

```bash
C=$WORK/cpu-mlir
git -C $C checkout -- qcom_arm64_backend && git -C $C clean -fdq qcom_arm64_backend/lib qcom_arm64_backend/test/Transforms/arm64-reduction-scheduling.mlir
git -C $C checkout -b sme-ext-test-B origin/main      # (or 12579ba, as in phase A)
git -C $C apply --whitespace=nowarn $KIT/patches/combined_with_pr14.patch
rebuild B
```
`git clean` only touches untracked files under `lib/` and the PR's new lit test; check
`git -C $C status --short` before it and report anything else untracked there instead of deleting it.

### 3.1 Test suite with patch B, default config (PR #14 on, SSVE off)

```bash
ENV=$ENV RUN_ON_SIM=1 bash $KIT/tools/sme_ext/regress.sh $RES/B_qemu
cat $RES/B_qemu/summary.txt
grep -E "FAIL|PASS" $RES/B_qemu/lit.log | grep -iE "reduction-scheduling|softmax"     # PR #14's own lit tests: never run by the author
```

### 3.2 Reductions in streaming mode: correctness (qemu at two SVLs, then the phone)

```bash
P=$ARM64_BACKEND_PATH/python
TRI="test_sum_reduce_triton.py triton_shared_examples/test_softmax.py triton_shared_examples/test_reduce.py triton_shared_examples/test_layernorm.py"
TOR="test_softmax_torch.py test_add_matmul_softmax.py test_sum_reduce_torch.py test_layernorm_torch.py"
for svl in 16 64; do Q="max,sme_fa64=off,sve-default-vector-length=16,sme-default-vector-length=$svl"; for fix in pr14 pr14fix; do
  F=$([ $fix = pr14fix ] && echo "TRITON_ARM64_REDSCHED_UNMASK=1")
  (cd $P/examples   && env $F TRITON_ARM64_SSVE_REDUCTIONS=1 QEMU_CPU=$Q RUN_ON_SIM=1 python -m pytest -q --target sme $TRI -p no:cacheprovider > $RES/ssve_${fix}_svl${svl}_triton.log 2>&1)
  (cd $P/torch-mlir && env $F TRITON_ARM64_SSVE_REDUCTIONS=1 QEMU_CPU=$Q RUN_ON_SIM=1 python -m pytest -q --target sme $TOR -p no:cacheprovider > $RES/ssve_${fix}_svl${svl}_torch.log 2>&1)
  echo "svl=$((svl*8)) $fix: triton $(tail -1 $RES/ssve_${fix}_svl${svl}_triton.log) | torch $(tail -1 $RES/ssve_${fix}_svl${svl}_torch.log)"
done; done | tee $RES/ssve_qemu_summary.txt
```
Expected: `pr14` (as posted) **fails** with SSVE (`LLVM ERROR: Expanding reductions for scalable
vectors is undefined`, about 10 Triton + 10 torch); `pr14fix` passes everything (43 Triton, 13 torch)
at both SVLs. Then on the phone (only if `sme=1`):
```bash
(cd $P/examples   && TRITON_ARM64_REDSCHED_UNMASK=1 TRITON_ARM64_SSVE_REDUCTIONS=1 RUN_ON_SIM=0 python -m pytest -q --target sme $TRI -p no:cacheprovider > $RES/ssve_phone_triton.log 2>&1); tail -1 $RES/ssve_phone_triton.log
(cd $P/torch-mlir && TRITON_ARM64_REDSCHED_UNMASK=1 TRITON_ARM64_SSVE_REDUCTIONS=1 RUN_ON_SIM=0 python -m pytest -q --target sme $TOR -p no:cacheprovider > $RES/ssve_phone_torch.log 2>&1); tail -1 $RES/ssve_phone_torch.log
```

### 3.3 Softmax timing on the phone: PR #14 variants and SSVE

```bash
ENV=$ENV RUN_ON_SIM=0 REPS=5 ITERS=20 bash $KIT/tools/sme_ext/device/device_bench.sh $RES/B_bench softmax
python $KIT/tools/sme_ext/device/summarize.py $RES/B_bench
```
Variants: `main` (PR #14 off), `pr14`, `pr14fix`, `reassocfix` on sve and sme; `ssve_pr14fix`,
`ssve_reassocfix` on sme only. Sizes: Triton 4x8192, 64x8192, 256x2048; torch 64x8192.
`TARGETS=sve` if no SME. The questions it answers:
1. Does PR #14 as posted beat `main` on the phone? (llvm-mca: ~17 % on the sum loop.)
2. Does the fix beat PR #14? Is `pr14fix` (unroll) any better than `reassocfix` (no unroll)?
3. Does SSVE (`ssve_*`) beat the same variant without it on sme, at this phone's SVL?

### 3.4 Optional: the asm and llvm-mca view

```bash
ENV=$ENV bash $KIT/tools/sme_ext/ab_reduction.sh $RES/B_mca triton 4 8192 sve    # per variant: match, sum-loop instructions, llvm-mca cycles/iteration
```
Expected cycles/iteration: main 12.1, pr14 10.0, pr14fix 3.2, reassocfix 3.1.

## 4. Optional: true main as a control

`base` is main's behaviour via switches. To check that claim on one size:
```bash
git -C $C checkout -- qcom_arm64_backend && git -C $C clean -fdq qcom_arm64_backend/lib qcom_arm64_backend/test/Transforms/arm64-reduction-scheduling.mlir
git -C $C checkout --detach origin/main && rebuild main
ENV=$ENV RUN_ON_SIM=0 REPS=5 SIZES="512 512 512" VARIANTS=all bash $KIT/tools/sme_ext/device/device_bench.sh $RES/main_bench matmul
```
(`all` on real main = main.) Its timing should match `base` from 2.3 within noise.

## 5. Report

Write `$RES/REPORT.md` with these sections, in this order. Numbers only from the files above; no
guesses. Mark anything you couldn't run as "not run" with the reason.

1. **Setup**: `original_branch.txt`, `origin_main.txt`, whether 12579ba was needed, `probe.txt`
   (sve/sme/fa64, VL, SVL), the CPU model from `cpuinfo.txt`.
2. **Phase A tests**: `A_qemu/summary.txt`, the four "expect" lines of 2.1, `A_phone/summary.txt`,
   every failure with its log excerpt and whether `base` fails too.
3. **Phase A timing**: `A_bench/RESULTS.md` as is, then one line per size: "matmul MxKxN sme: X ms -> Y ms (Z x)".
4. **Phase B tests**: `B_qemu/summary.txt`, PR #14's lit test lines, `ssve_qemu_summary.txt`, the phone SSVE lines.
5. **Phase B timing**: `B_bench/RESULTS.md` as is, then answers to the three questions in 3.3, each with the numbers.
6. **Optional runs** (2.4, 3.4, 4) if done.
7. **Anomalies**: anything unexpected (variance, SIGILL, a variant that didn't change the asm: compare the
   `smstart in asm` column and `grep -c fadda` / `faddv` in `raw/**/*-arm64.S`).

Then pack the results without the executables and raw tensors:
```bash
cd $WORK && tar --exclude='*.exe' --exclude='*.raw' --exclude='*.qemu.log' -czf sme_ext_results.tar.gz sme_ext_results && ls -lh sme_ext_results.tar.gz
```

## 6. Interpreting (for the report's conclusions)

- Patch A is worth a PR if `all` is faster than `base` on sme at every size (and not slower on sve).
  Expected: a larger gain than qemu's 24 %, because 2600 mode switches per call are gone.
- PR #14: if `pr14` ≈ `main`, the unroll alone doesn't help (as predicted, since `fadda` stays). If
  `reassocfix` ≈ `pr14fix`, the unroll isn't needed once `reassoc` reaches the asm.
- SSVE: worth pursuing only if `ssve_*` beats its non-SSVE twin on sme. On qemu it was +4 % worse
  at SVL 128 and 15 % fewer instructions at SVL 512, so the probe's SVL is the key context.

## 7. Restore the checkout

```bash
C=$WORK/cpu-mlir
git -C $C checkout -- qcom_arm64_backend && git -C $C clean -fdq qcom_arm64_backend/lib qcom_arm64_backend/test/Transforms/arm64-reduction-scheduling.mlir
git -C $C checkout $(head -1 $RES/original_branch.txt) && rebuild restore
git -C $C status --short | grep -v '^??'      # must match status_before.txt
```
Leave the `sme-ext-test-*` branches (they hold no commits; `git branch -D` them only if asked).
