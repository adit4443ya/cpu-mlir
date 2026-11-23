# Extending the SME outlining: matmul packing, SSVE reductions, PR #14

Written 2026-11-22, after the softmax PR was merged upstream (main = `12579ba`).
Everything here runs on the laptop under qemu. **qemu counts instructions, it
does not model time**: the cycle numbers come from `llvm-mca` (Neoverse-V2,
non-streaming), and none of this has been timed on the device yet.

## 0. State and setup

| what | where |
|---|---|
| work branch (uncommitted, not pushed) | `branch/cpu-mlir` on `sme-outline-ext` = `origin/main` 12579ba + working-tree changes |
| my changes only, as a patch | `runs/sme_ext/sme-outline-ext.patch` |
| PR #14 as posted (jsarrao, `df26047`) | `runs/sme_ext/pr14.patch` (fetched with `git fetch origin '+refs/pull/*/head:refs/remotes/origin/pr/*'`) |
| my changes + PR #14 + its suggested fix + measurement switches | `runs/sme_ext/combined_with_pr14.patch` |
| device runbook + kit for the dev-compute agent | `docs/SME_EXT_DEVICE_RUNBOOK.md`; kit: `bash tools/sme_ext/device/make_kit.sh` -> `runs/sme_ext/sme_ext_kit.tar.gz` |
| tools | `tools/sme_ext/`: `qprof.py`, `mca_loop.py`, `survey.py`, `ab_matmul.sh`, `ab_reduction.sh`, `run_matmul.py`, `regress.sh` |
| outputs | `runs/sme_ext/` |

```bash
cd ~/thats_me/triton-mlir-arm
source env-branch.sh
export QEMU_CPU=max,sme_fa64=off        # qemu's default `max` has FA64, which hides streaming-mode bugs
bash build.sh branch build              # after any C++ change (or after applying a patch)
lit -q $TRITON_BUILD_DIR/third_party/qcom_arm64_backend/test   # a.mlir/b.mlir = your scratch, UNRESOLVED
```

To measure with one build, the working tree has **temporary env switches** (marked
`TEMPORARY` in the code; they must be removed before a PR):

| env var | turns off / on |
|---|---|
| `TRITON_ARM64_NO_PACK_HOIST=1` | off: the accumulator pack/unpack hoist (§1.2) |
| `TRITON_ARM64_NO_OUTLINE_RESULTS=1` | off: outlining loops that have results (§1.1) |
| `TRITON_ARM64_NO_BUFFER_HOIST=1` | off: BufferLoopHoisting (§1.3) |
| `TRITON_ARM64_SSVE_REDUCTIONS=1` | on: outline reduction loops into streaming functions (§3) — experimental, default off |
| with `combined_with_pr14.patch` only: `TRITON_ARM64_NO_REDUCTION_SCHEDULING=1`, `TRITON_ARM64_REDSCHED_REASSOC_ONLY=1`, `TRITON_ARM64_REDSCHED_UNMASK=1` | PR #14 off / reassoc without unrolling / the suggested fix (§4) |

All three of §1 together = main's behaviour:
`TRITON_ARM64_NO_PACK_HOIST=1 TRITON_ARM64_NO_OUTLINE_RESULTS=1 TRITON_ARM64_NO_BUFFER_HOIST=1`.

## 1. Triton matmul on SME: packing inside the k-loop, and why the whole kernel was streaming

### What main does

`test_matmul.py` (BLOCK 32x64x16): `tl.dot` sits inside the k-loop, so
MatmulToMmt4D block-packs **every iteration**:

```bash
cd runs/sme_ext/mm_tr          # the kernel's -ttshared.mlir is there (made by run_matmul.py with SAVE_TEMPS)
TRITON_ARM64_NO_PACK_HOIST=1 arm64-mlir-opt --vector-ext=sme -optimize-triton-shared \
  -mlir-print-ir-after=triton-arm64-matmul-to-mmt4d matmul_kernel-ttshared.mlir -o /dev/null 2>&1 \
  | grep -E "scf.for|linalg.(mmt4d|transpose|copy)|scf.yield"
```
```
scf.for ... iter_args(..., %arg18 = %1) -> (index, index, tensor<32x64xf32>)
  linalg.transpose (A 32x16 -> 1x16x32x1)        pack A   (needed: new tile each k)
  linalg.transpose (B 16x64 -> 2x16x32x1)        pack B   (needed)
  linalg.transpose (C 32x64 -> 1x2x32x32)        pack C   <- every iteration
  linalg.mmt4d
  linalg.transpose (C back) + collapse + copy    unpack C <- every iteration
```

Then `ARM64SMEFuncOutlinePass` **refused the k-loop** (`outlineLoop`: "After
bufferization the loop writes to memrefs and has no results") — but the Triton
k-loop carries two `index` pointer offsets as results. So the ZA code stayed in
`@matmul_kernel` and EnableArmStreaming made the **whole kernel** streaming.
Inside that streaming code, every k-iteration also `malloc`s and `free`s the
two masked-load buffers (A 32x16, B 16x64): each call is `smstop sm` / call /
`smstart sm` plus a lazy ZA save (`__arm_tpidr2_save/restore`).

### 1.1 Outline loops with results (`ARM64SMEFuncOutlinePass.cpp`)
The outlined function returns the loop's results and the call replaces them.
Scalable vectors still may not cross the boundary (args or results); fixed-size
ones may. Needed one launcher fix: `doesFuncReturnValue`
(`LowerTritonLinalgToLLVM.cpp`) now only looks at public functions, otherwise
a private helper returning values made the Triton kernel get torch-style C
wrappers and `get_kernel_name` asserted.

### 1.2 Hoist the accumulator pack/unpack (`ARM64MatmulToMmt4DPass.cpp`)
`hoistPackedAccumulator`: for an `scf.for` iter_arg whose only uses are
`linalg.pack %acc` and the destination of the `linalg.unpack` that is yielded,
carry the packed tensor instead: pack the init before the loop, unpack the
result after it. Runs between block packing and pack lowering (the pass now
applies its patterns in two rounds). Refused with padding (unpack drops padded
rows, repack refills them), dynamic shapes, or mismatching tiles/perms.

### 1.3 Allocate per-iteration buffers once (`OptimizeTritonShared.cpp`)
`bufferization::createBufferLoopHoistingPass()` before the dealloc pipeline:
static-size allocs in a loop move before it, so no malloc/free (and no mode
switch) per k-iteration. Helps every target, not only SME.

### Results (Triton matmul 179x167x321, SME, SVL 128, qemu FA64 off)

```bash
bash tools/sme_ext/ab_matmul.sh runs/sme_ext/mm            # 5 variants, all "match: True"
# optional size: bash tools/sme_ext/ab_matmul.sh runs/sme_ext/mm256 256 256 256
```

| variant | kernel instructions | `smstart` executed | public kernel streaming? |
|---|---:|---:|---|
| base (= main) | 41.92 M | 2772 | yes, all of it |
| + pack hoist | 31.87 M (**-24 %**) | 2844 | yes |
| + outline only | 41.69 M | 36 + 2484 | no |
| hoist + outline | 31.87 M | 36 + 2484 | no |
| **all three** | **31.88 M** | **36 + 108** | **no** |

(36 = one call per program; the 108 left are `smstart za` around the
streaming-compatible `__arm_sc_memset`, no `smstart sm`.) The instruction win is
the hoist; outlining + buffer hoisting remove ~2600 mode switches, which qemu
cannot price — on hardware each is a pipeline flush plus spills, so expect the
device gain to be larger than the instruction count says.

Check the streaming function no longer calls malloc:
```bash
awk '/^matmul_kernel_sme_0:/,/^.Lfunc_end/' runs/sme_ext/mm/all/matmul_kernel-arm64.S | grep -E "\bbl\b|smstart|smstop" | sort | uniq -c
```

## 2. Other cases: survey of all 208 collected test kernels

`runs/specialize/inputs/{triton,torch}` = every unique pipeline input of the
Triton and torch suites (collected for the specialization study).

```bash
cd runs/sme_ext/survey
O=$(which arm64-mlir-opt); I=../../specialize/inputs
TRITON_ARM64_NO_PACK_HOIST=1 TRITON_ARM64_NO_OUTLINE_RESULTS=1 TRITON_ARM64_NO_BUFFER_HOIST=1 \
  python ../../../tools/sme_ext/survey.py $O base.json $I/triton $I/torch      # main
python ../../../tools/sme_ext/survey.py $O new.json $I/triton $I/torch           # this branch
```

| | main | this branch |
|---|---:|---:|
| kernels with ZA code outlined (`_sme_N`) | 21 | 23 |
| public function streaming as a whole | **2** (Triton matmul, flash attention) | **0** |
| allocs / copies / calls inside streaming code | 5 | 1 |

Left over, not fixed here:
- **Flash attention** (`attention_fwd_kernel_sme_0`): bufferization copies the
  loop-carried buffers every iteration (`memref.copy` of 8x2x32x32 and 256x64
  f32, 64 KB each). These become memcpy calls, which LLVM makes
  streaming-compatible (`__arm_sc_memcpy`), so there is no mode switch, but the copies are pure overhead. The
  cause is non-in-place iter_args in bufferization; worth its own look.
- **Triton matmul output store**: `memrefCopy` (runtime, 0.9 M of the 31.9 M
  instructions) for the masked C store, a dynamic strided subview the softmax
  PR's `convertStridedCopiesToLinalg` doesn't catch.
- **Keep C in ZA across k**: each k-iteration still loads the packed C tile into
  ZA and stores it back (mmt4d per iteration). Now that the whole k-loop is one
  streaming function, the next step is to keep the accumulator in ZA tiles
  across iterations (zero before the loop, mopa per k, store after).
- **torch matmuls** have no loop around them: pack/unpack run once per call.
  With constant weights (GPT-2 etc.), packing B could be constant-folded at
  compile time; and the elementwise-fusion regression still keeps GPT-2's matmuls
  off mmt4d.

## 3. Reductions in streaming SVE (SSVE)

Idea: on SME, reductions run as normal SVE at the SVE vector length. In
streaming mode the vector length is the streaming one (SVL), which can be
larger. `ssve-reductions` (option of `triton-arm64-sme-func-outline`, env
`TRITON_ARM64_SSVE_REDUCTIONS=1`) outlines every outermost loop with a scalable
`vector.reduction` into `<fn>_ssve_N` with `llvm.arm_locally_streaming`.

Three things it has to respect (all found by running it):

1. **In-order fp sums are illegal in streaming mode.** `fadda` is not a
   streaming instruction and LLVM cannot expand it for scalable vectors:
   ```bash
   cd runs/sme_ext/llc
   llc -mtriple=aarch64 -mattr=+v8.5a,+sve2,+sme red.ll -o /dev/null      # LLVM ERROR: Expanding reductions for scalable vectors is undefined.
   llc -mtriple=aarch64 -mattr=+v8.5a,+sve2,+sme red_no_ordered.ll -o - | grep -E "faddv|fmaxnmv"   # reassoc -> faddv, max -> fmaxnmv: fine
   ```
   So loops with an fp add/mul reduction without `reassoc` are not outlined.
   On main that leaves only the softmax **max** loops (3 kernels); with
   reassoc sums (PR #14 + the fix in §4) it is 20 kernels.
2. **Gathers/scatters** are illegal in streaming mode without FA64: refused.
3. **`arm_locally_streaming`, not `arm_streaming`.** With `arm_streaming` the
   caller sets up the arguments after `smstart sm`, and LLVM materialized a
   `<1 x float>` zero argument with NEON `movi v0.2d, #0` there: SIGILL in 3
   torch sum tests (`_mlir_ciface_SumReduce`, see `runs/sme_ext/sigill/dis.txt`).
   Locally streaming switches inside the callee, so the caller's code is normal.

Results (Triton softmax 4x8192, SME, qemu instruction counts):

```bash
# needs the combined patch (PR #14 + fix) for the sum loop; see §4 for applying it
for svl in 16 64; do Q="max,sme_fa64=off,sve-default-vector-length=16,sme-default-vector-length=$svl"
  QEMU_CPU=$Q VARIANTS=reassocfix QPROF=$Q bash tools/sme_ext/ab_reduction.sh runs/sme_ext/red_sme_$svl/off triton 4 8192 sme
  QEMU_CPU=$Q SSVE=1 VARIANTS="pr14fix reassocfix" QPROF=$Q bash tools/sme_ext/ab_reduction.sh runs/sme_ext/red_sme_$svl/on triton 4 8192 sme
done
```

| | max loop | sum loop | rest of kernel | kernel total |
|---|---:|---:|---:|---:|
| SVE (VL 128), no SSVE | in kernel | in kernel | | 594 K |
| SSVE, SVL 128 | 86 K | 66 K | 467 K | 619 K (+4 %) |
| SSVE, SVL 512 | 21.5 K | 16.4 K | 467 K | 505 K (-15 %) |

Correctness, SSVE on (with PR #14 + fix), both SVL 128 and SVL 512: Triton
softmax/sum_reduce/reduce/layernorm **43/43**, torch
softmax/add_matmul_softmax/sum_reduce/layernorm **13/13**.

What it means: SSVE only pays when **SVL > SVE VL** *and* the streaming unit's
throughput is good — on some cores (e.g. Apple M4) streaming SVE ops are much
slower than the core's own NEON/SVE. It is a per-device decision, so it stays an
opt-in experiment until we know the device's SVL and time it there. The
remaining 467 K (exp and divide loops) would move too if we outlined whole
row loops instead of only the reductions.

## 4. PR #14 (jsarrao): "add reduction scheduling pass"

What it does: after vector lowering, a loop of the exact shape
`extract(acc) -> [vector.mask] vector.reduction <add> -> broadcast -> yield`
on a scalable float vector is unrolled 4x with 4 independent accumulators,
the extra copies guarded by `scf.if (iv + k*step < ub)`, combined after the
loop; every reduction and combine gets `fastmath<reassoc>`. SVE/SME only.

Apply it on top of this branch (it conflicts on one header line with my patch):
```bash
cd branch/cpu-mlir
git checkout -- qcom_arm64_backend && git apply --whitespace=nowarn ../../runs/sme_ext/combined_with_pr14.patch   # mine + PR + fix + switches
cd ../.. && bash build.sh branch build
bash tools/sme_ext/ab_reduction.sh runs/sme_ext/red_sve triton 4 8192 sve     # 4 variants + llvm-mca
# back to my changes only:
cd branch/cpu-mlir && git checkout -- qcom_arm64_backend && rm -f qcom_arm64_backend/lib/OptimizeTritonShared/ARM64ReductionSchedulingPass.cpp qcom_arm64_backend/test/Transforms/arm64-reduction-scheduling.mlir \
  && git apply ../../runs/sme_ext/sme-outline-ext.patch && cd ../.. && bash build.sh branch build
```

### Finding 1 (main one): `reassoc` never reaches LLVM for masked reductions
The tiled reductions are always masked (`vector.mask { vector.reduction }`).
Upstream `MaskedReductionOpConversion` (mlir/lib/Conversion/VectorToLLVM/ConvertVectorToLLVM.cpp)
lowers them to `llvm.intr.vp.reduce.fadd` — it computes `fmf` but the LLVM
dialect's VP reduction ops have no fast-math attribute
(`LLVM_VPReductionBase` in LLVMIntrinsicOps.td), so the flag is dropped and
LLVM emits the in-order `fadda` anyway:
```bash
grep -c "vp.reduce.fadd" runs/sme_ext/ssve_r1/sum_reduce_rank1_kernel-llvm-dialect.mlir   # no fastmath on them
grep -cE "\bfadda\b" runs/sme_ext/ssve_r1/sve/*.S     # 8 with PR #14 on SVE
grep -cE "\bfaddv\b" runs/sme_ext/ssve_r1/sve/*.S     # 0
```
The PR's lit test only checks `fastmath<reassoc>` in MLIR, so it doesn't see this.
Suggested fix (in `combined_with_pr14.patch`, switch `TRITON_ARM64_REDSCHED_UNMASK=1`):
rewrite a masked reassoc fp-add reduction to `select(mask, v, 0)` + unmasked
`vector.reduction ... fastmath<reassoc>`, which lowers to
`llvm.vector.reduce.fadd reassoc` -> `faddv`. Proper fix upstream: give the VP
reduction ops fast-math flags (an MLIR patch we could send).

### Finding 2: with reassoc working, the 4x unroll adds nothing
llvm-mca, Neoverse-V2, the sum loop of Triton softmax 4x8192 on SVE
(`python tools/sme_ext/mca_loop.py runs/sme_ext/red_sve/<variant>/softmax_kernel-arm64.S`).
One original iteration = one `vector<[8]xf32>` = 2 Z registers.

| variant | sum loop code | cycles per original iteration |
|---|---|---:|
| main | 2x `fadda` (one chain) | 12.1 |
| PR #14 as posted | 4 chains of `fadda` | 10.0 (-17 %) |
| PR #14 + fix | 4x `faddv` + `fadd` | 3.2 (-74 %) |
| reassoc + fix, **no unroll** | `fadd` + `faddv` | **3.1 (-74 %)** |
| vector accumulator (hand-written IR, `runs/sme_ext/llc/vacc.ll`) | 2x `fadd z` per iteration, one `faddv` after the loop | 2.4 |

With `faddv` the loop-carried chain is a single scalar `fadd`, so splitting it
4 ways doesn't help; the unroll only adds code and `scf.if` guards. The best
form is a **vector accumulator** (add whole vectors in the loop, reduce once
after it) — that is a vectorization-side change (e.g. reduction tiling that
keeps a partial vector, `scf::tileReductionUsingFor`-style), not a post-pass.

### Finding 3: with SSVE, PR #14 as posted crashes llc
Because of finding 1 the sum is still `fadda`, illegal in streaming mode:
`LLVM ERROR: Expanding reductions for scalable vectors is undefined` (10 Triton
+ 10 torch failures with SSVE on). Without SSVE (the PR's own scope) the sum
tests pass on sme and sve (11/11 each, `test_sum_reduce_triton.py` + `test_layernorm.py`).

### Smaller comments
- `rewriter.create<Op>(...)` vs `Op::create(rewriter, ...)` mixed in one file (the latter is the current MLIR style; `scf::ForOp::create` is already used).
- `if (kAccumulatorCount < 2) return failure(); // bail out (for testing)` is dead code with a constexpr 4.
- Zero init `+0.0` for partial sums: fine with reassoc; -0.0 is the exact identity.
- Not checked: the PR's own lit tests with my patch on top (I ran its pytest cases, not `lit`).

### What I would reply on the PR
1. The reassoc doesn't survive the masked lowering (asm still `fadda`); please check the asm / add an llc-level check.
2. With reassoc working (select + unmasked reduction), llvm-mca shows the same win without unrolling; suggest dropping the unroll, or measuring on the device whether it adds anything.
3. Longer term: vector accumulators in the vectorizer instead of a post-pass.

## 5. Regression status (this branch, without PR #14, default config)

```bash
bash tools/sme_ext/regress.sh runs/sme_ext/regress       # Triton + torch suites, sme and sve, FA64 off (~1 h)
cat runs/sme_ext/regress/summary.txt
```
Results 2026-11-22 (this branch, default config, QEMU_CPU=max,sme_fa64=off):

| suite | sme | sve |
|---|---|---|
| Triton (`python/examples`) | 195 passed | 195 passed |
| torch (`python/torch-mlir`) | 20 passed | 20 passed |
| lit | 15/15 (+ a.mlir/b.mlir scratch unresolved) | |

The new lit cases fail with the switches that undo them (`TRITON_ARM64_NO_PACK_HOIST=1`, `TRITON_ARM64_NO_OUTLINE_RESULTS=1`).

## 6. Next steps (my suggestion)

1. PR A (small, clear win): §1.1 + §1.2 + §1.3 + lit tests, without the env switches.
2. Device timing of the Triton matmul: main vs PR A (mode switches are what qemu can't price).
3. Keep C in ZA across the k-loop (§2).
4. Reply on PR #14 with §4.
5. SSVE: get the device SVL first; only then decide whether `ssve-reductions` is worth a PR.
7. Device run: `docs/SME_EXT_DEVICE_RUNBOOK.md` (for an agent on dev compute; gives the phone timings for §1, §3 and §4).

## 7. How this was approached, step by step (every command)

The order matters more than any single step: **measure, find the cause in the IR, check how
general it is, change one thing behind a switch, test, measure again.** All from
`~/thats_me/triton-mlir-arm` with `source env-branch.sh` and `export QEMU_CPU=max,sme_fa64=off`.

**Step 1: profile the kernel as it is.** Where do the instructions and mode switches go?
```bash
VARIANTS=base bash tools/sme_ext/ab_matmul.sh runs/sme_ext/mm      # temps + exe in runs/sme_ext/mm/base
python tools/sme_ext/qprof.py runs/sme_ext/mm/base/matmul_kernel.exe   # per symbol: kernel, malloc/free, memrefCopy, smstart count
grep -cE "^\s+smstart" runs/sme_ext/mm/base/matmul_kernel-arm64.S
grep -n "aarch64_pstate_sm_enabled\|arm_streaming" runs/sme_ext/mm/base/matmul_kernel-llvm-ir.ll | head   # is the PUBLIC kernel streaming?
```
Finding: 2772 `smstart`, malloc/free inside streaming code, and the whole public kernel streaming.

**Step 2: find the cause in the IR, pass by pass.**
```bash
cd runs/sme_ext/mm/base          # on this branch, set the switches to see main's behaviour:
export TRITON_ARM64_NO_PACK_HOIST=1 TRITON_ARM64_NO_OUTLINE_RESULTS=1 TRITON_ARM64_NO_BUFFER_HOIST=1
arm64-mlir-opt --vector-ext=sme -optimize-triton-shared -mlir-print-ir-after=triton-arm64-matmul-to-mmt4d \
  matmul_kernel-ttshared.mlir -o /dev/null 2>&1 | grep -E "scf.for|linalg.(pack|unpack|mmt4d|transpose|copy)|scf.yield"
# -> pack C / unpack C inside the k-loop (§1)
arm64-mlir-opt --vector-ext=sme -optimize-triton-shared -debug-only=triton-arm64-sme-func-outline \
  matmul_kernel-ttshared.mlir -o /dev/null 2>&1 | grep "sme-func-outline"
# -> "[triton-arm64-sme-func-outline] could not outline: %37:2 = scf.for ... -> (index, index)": the loop has results
arm64-mlir-opt --vector-ext=sme -optimize-triton-shared -mlir-print-ir-after-all -mlir-disable-threading \
  matmul_kernel-ttshared.mlir -o /dev/null 2> after_all.mlir
grep -n "IR Dump After\|memref.alloc\|memref.dealloc" after_all.mlir | less   # where the per-iteration allocs appear (bufferization)
unset TRITON_ARM64_NO_PACK_HOIST TRITON_ARM64_NO_OUTLINE_RESULTS TRITON_ARM64_NO_BUFFER_HOIST
```

**Step 3: check how general it is before fixing it.** One kernel is an anecdote; the survey runs the
pipeline on all 208 collected test inputs and reports streaming placement per kernel:
```bash
cd runs/sme_ext/survey; O=$(which arm64-mlir-opt); I=../../specialize/inputs
TRITON_ARM64_NO_PACK_HOIST=1 TRITON_ARM64_NO_OUTLINE_RESULTS=1 TRITON_ARM64_NO_BUFFER_HOIST=1 \
  python ../../../tools/sme_ext/survey.py $O base.json $I/triton $I/torch
```
Finding: 2 public kernels streaming as a whole (Triton matmul, flash attention), 5 with calls in streaming code.

**Step 4: change one thing at a time, each behind a temporary env switch.** One build then
produces every variant (`ab_matmul.sh` runs base/hoist/outline/both/all), so each change's effect is
measured alone and combined. Order chosen: outline-with-results (unblocks everything), the pack
hoist (biggest instruction win), BufferLoopHoisting (upstream pass, one line). After each:
```bash
bash build.sh branch build
bash tools/sme_ext/ab_matmul.sh runs/sme_ext/mm          # every variant must print "match: True"
```

**Step 5: lit test per change, and prove it tests the change.** Add the case, then run its RUN line
with the switch that undoes the change: FileCheck must fail.
```bash
S=branch/cpu-mlir/qcom_arm64_backend/test/Transforms
TRITON_ARM64_NO_PACK_HOIST=1 arm64-mlir-opt $S/arm64-matmul-to-mmt4d-pass.mlir -split-input-file -triton-arm64-matmul-to-mmt4d | FileCheck $S/arm64-matmul-to-mmt4d-pass.mlir   # must FAIL
arm64-mlir-opt $S/arm64-matmul-to-mmt4d-pass.mlir -split-input-file -triton-arm64-matmul-to-mmt4d | FileCheck $S/arm64-matmul-to-mmt4d-pass.mlir                                # must pass
```

**Step 6: fix what the new behaviour breaks downstream.** Outlining a loop with results made a
private function that returns values; the launcher's `doesFuncReturnValue` then treated the Triton
kernel like a torch function and `get_kernel_name` asserted. Found by running the suite; fixed by
looking at public functions only. (Rule: before adding functions, grep `lib/LowerTritonLinalgToLLVM`
and `backend/*.py` for code that walks every function.)

**Step 7: re-survey and run the whole suite.**
```bash
cd runs/sme_ext/survey && python ../../../tools/sme_ext/survey.py $(which arm64-mlir-opt) new.json ../../specialize/inputs/triton ../../specialize/inputs/torch; cd -
bash tools/sme_ext/regress.sh runs/sme_ext/regress && cat runs/sme_ext/regress/summary.txt
```

**Step 8 (SSVE): hypothesis -> option -> run -> each failure to a minimal repro -> rule.**
- Added `ssve-reductions` (default off) and ran the reduction tests with FA64 off.
- `llc` crashed: cut the kernel's `.ll` down to one reduction (`runs/sme_ext/llc/red.ll`), ran
  `llc -mattr=+v8.5a,+sve2,+sme red.ll`: ordered `fadd` reduction in a streaming function is not
  lowerable. Rule: refuse loops with ordered fp reductions.
- SIGILL in 3 torch tests: `qemu-aarch64-static -cpu max,sme_fa64=off -d in_asm,nochain -D q.log ./SumReduce.exe`,
  last block in `q.log`, looked up in `llvm-objdump -d --mattr=+sve2,+sme SumReduce.exe` -> NEON `movi`
  after `smstart sm` in the caller (`runs/sme_ext/sigill/dis.txt`). Rule: `arm_locally_streaming`.
- Measured at two SVLs (§3) because the answer depends on SVL.

**Step 9 (PR #14 review): apply, run its tests, then read the asm, not only the IR.**
```bash
git -C branch/cpu-mlir fetch origin '+refs/pull/*/head:refs/remotes/origin/pr/*'
git -C branch/cpu-mlir diff origin/main origin/pr/14 > runs/sme_ext/pr14.patch      # then apply as in §4
grep -cE "\bfadda\b|\bfaddv\b" runs/sme_ext/red_sve/pr14/*-arm64.S     # the IR said reassoc, the asm still had fadda
grep -n "MaskedReductionOpConversion" -A40 shared/llvm-project/mlir/lib/Conversion/VectorToLLVM/ConvertVectorToLLVM.cpp | grep -n "fmf\|VPReduce"
python tools/sme_ext/mca_loop.py runs/sme_ext/red_sve/pr14/softmax_kernel-arm64.S                 # cycles/iteration
```
Then wrote the smallest fix that makes the flag survive (select + unmasked reduction), and measured
every combination (`ab_reduction.sh`) to separate "reassoc" from "unroll".

