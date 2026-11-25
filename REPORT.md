# SME outline extension and PR #14: full results

All numbers below are read from files under `sme_ext_results/` (the generator is `sme_ext_tools/make_report.py`). Times from the executable's own benchmark are **ms for 20 iterations** unless a table says otherwise. Speedup = baseline median / variant median, so above 1 means faster than the baseline. Each section names its session; numbers from different sessions are never mixed in one ratio.

## Contents
- Summary
- 1. Setup
- 2. Phase A tests (patch A)
- 3. Phase A timing (Canoe phone)
- 4. Phase B tests (combined patch)
- 5. Phase B timing (Canoe phone)
- 6. Optional runs: main control, llvm-mca, three-build matrix, HAWI
- 7. Anomalies and caveats
- Appendix: raw result tables
- Appendix: files

## Summary

What the labels are (checked against the patch files and the compiled `libtriton.so`):

| Label | Build | Contents |
|---|---|---|
| main | origin/main, 12579ba | nothing added |
| A | main + `sme-outline-ext.patch` | three matmul changes (pack hoist, outlined loops that return values, buffer hoist) and an SSVE-reductions option that is off by default |
| B | main + `combined_with_pr14.patch` | everything in A, plus PR #14's reduction-scheduling pass (4 accumulators, `reassoc`), on by default |

A and B generate identical matmul code (LLVM IR diff: 0 changed lines at 1024³ on sme and sve), so B differs from A only on reductions and softmax.

Findings (each from the section named):

1. **Patch A speeds up sve matmul** by about 1.2x at 512³ and 1024³ (Canoe matrix: 1.176x and 1.221x; HAWI: 1.231x and 1.193x). Section 3 and 6.
2. **Patch A does not speed up sme matmul** on the phones (Canoe 512³ 1.001x, 1024³ 0.981x; HAWI 1024³ 1.001x), although it cuts the streaming-mode entries in the generated code (17 `smstart` to 4) and the qemu instruction count by about 24 % (hoists). The runbook expected a larger gain on the phone; it did not appear.
3. **B (PR #14) speeds up row-sum reductions by about 1.7x to 2.1x** on Canoe, e.g. triton 4x8192 sme 2.13x, sve 2.10x; torch 64x8192 sme 1.94x. Section 6.
4. **B speeds up Triton softmax by about 1.05x to 1.14x** (4x8192 sme 1.134x; 256x2048 sme 1.064x). Torch softmax 64x8192 is within noise (1.017x sme, 1.030x sve). Section 5 and 6.
5. **PR #14 as posted fails in streaming mode** (SSVE): 10 Triton + 10 torch tests fail on qemu at both vector lengths; the fixed variant (`pr14fix`) passes 43 Triton + 13 torch on qemu and on the phone. Section 4.
6. **Correctness:** every timed run on both phones matched its reference (Canoe matrix 360 of 360, HAWI 60 of 60). Test suites passed with no failures on qemu for A, B and main, and on the Canoe phone for A (full suite) and for the SSVE subset of B; on HAWI a trimmed set passed for A and main. Section 2, 4 and 6.
7. **Open points:** the sme 1024³ slowdown of A (0.981x) and B (0.956x) on Canoe is not explained by the code, because A and B have identical matmul code; the run order inside each rep is the likely cause, but this was not tested. Section 7.

## 1. Setup

| Item | Value |
|---|---|
| Starting branch (`original_branch.txt`) | `sme-o` |
| origin/main (`origin_main.txt`) | `12579ba Bump qualcomm/qcom-reusable-workflows/.github/workflows/reusable-qcom-preflight-checks-orchestrator.yml` |
| Was 12579ba needed? | Yes: all three builds are origin/main (12579ba) plus the patch. The patch-A tree matches `sme-o` for `qcom_arm64_backend` (`phone_progress.log`: 'tree sme-ext-test-A == sme-o'; no `A_vs_sme-o.diff` was written). |
| Canoe phone | `hyd-lablnx294` / `5a28e3c2`, product canoe, SoC model SM8850, 8 cores, CPU implementer 0x51, CPU part 0x002 |
| Canoe probe (`probe.txt`) | sve=1 sve2=1 sme=1 sme_fa64=0, sve_vl_bits=128, sme_svl_bits=512 |
| HAWI phone | `hyd-lablnx369` / `e8ab1b5`, product art, SoC model SM8975, 8 cores, CPU implementer 0x51, CPU part 0x003. I took this serial to be the device you called HAWI. |
| HAWI probe (`hawi/probe.txt`) | sve=1 sve2=1 sme=1 sme_fa64=0, sve_vl_bits=128, sme_svl_bits=512 |
| Other phones | `51822941` (on hyd-lablnx369) was not used. The earlier `33df9917` / `hyd-lablnx369` phone was not attached. |
| Driver edit | `_get_device_dir` in `driver.py` reads `TRITON_ARM64_DEVICE_DIR`, default unchanged. Uncommitted; re-applied after every checkout by `sme_ext_tools/apply_devdir.py`; still present in the final checkout. Canoe and HAWI both used `/data/local/tmp`, with separate host staging directories. |
| Environment wrapper | `cpu-mlir-env.sh` forces `RUN_ON_SIM=1`; `env_run.sh` sets it from `KIT_RUN_ON_SIM` (0 = phone). All phone runs used it. |
| Final checkout | `sme-o` @ b0b925a, rebuilt. `git status` (tracked changes, `status_after.txt`): `driver.py` modified, submodules `triton` and `triton-shared` marked modified. `status_before.txt` was never recorded, so I could not compare against it; this matches the expected pattern (driver edit + submodule pointers). |
| Build fingerprints (strings in `libtriton.so`) | main: `NO_PACK_HOIST`=0 and `NO_REDUCTION_SCHEDULING`=0; A: pack>=1, red=0; B: both >=1 (see matrix `progress.log` / `resume_progress.log`) |

## 2. Phase A tests (patch A)

### 2.1 qemu, A (`A_qemu/summary.txt`)
```
lit:  Passed : 15 (93.75%)  Unresolved: 1 (6.25%)
triton sme: 195 passed, 8254 skipped, 74 warnings in 557.53s (0:09:17)
torch  sme: 21 passed in 245.84s (0:04:05)
triton sve: 195 passed, 8254 skipped, 74 warnings in 561.93s (0:09:21)
torch  sve: 21 passed in 244.99s (0:04:04)
```
The one unresolved lit item is a scratch file `a.mlir` in the test directory, not a test failure; it is also unresolved on main (below).

The four switch checks (`A_new_lit_checks.txt`); each "expect" matched:
```
switch-off matmul lit: exit 1 (expect 1)
switch-off outline lit: exit 1 (expect 1)
default matmul lit: exit 0 (expect 0)
default outline lit: exit 0 (expect 0)
```

### 2.2 qemu, main control (`main_qemu/summary.txt`)
```
lit:  Passed : 15 (93.75%)  Unresolved: 1 (6.25%)
triton sme: 195 passed, 8254 skipped, 74 warnings in 549.01s (0:09:09)
torch  sme: 21 passed in 250.38s (0:04:10)
triton sve: 195 passed, 8254 skipped, 74 warnings in 563.46s (0:09:23)
torch  sve: 21 passed in 247.95s (0:04:07)
```

### 2.3 Canoe phone, A (`A_phone/summary.txt`)
```
lit:  Passed : 15 (93.75%)  Unresolved: 1 (6.25%)
triton sve: 195 passed, 8254 skipped, 74 warnings in 2172.69s (0:36:12)
torch  sve: 21 passed in 689.74s (0:11:29)
triton sme: 195 passed, 8254 skipped, 74 warnings in 2168.21s (0:36:08)
torch  sme: 21 passed in 680.41s (0:11:20)
```
There were no failures in A (qemu or phone), so there is no log excerpt and no question whether `base` fails too. (The `DEVICE_PATH lines 0` check in `phone_progress.log` is meaningless: pytest hides the output of passing tests. The separate smoke run did print `DEVICE_PATH: /data/local/tmp/...` and matched torch.)

### 2.4 qemu instruction counts, 179x167x321 sme (`A_qprof.stdout`)
Total instructions of the kernel per variant (one qemu run each; `base` = patch A's three switches all off):

| Variant | Kernel total | Streaming entries | Result |
|---|---:|---|---|
| base | 41,921,163 | 2772 | match=True |
| hoist | 31,868,409 | 2844 | match=True |
| outline | 41,693,823 | 2484+36 | match=True |
| both | 31,867,905 | 2484+36 | match=True |
| all | 31,884,495 | 108+36 | match=True |

`all` executes 23.9 % fewer instructions than `base` (41,921,163 to 31,884,495). The hoists account for it (`hoist` alone: 31,868,409); outlining alone barely changes the count (41,693,823) but cuts streaming-mode entries from 2772 to 108+36 once combined with the other changes.

### 2.5 Test counts
The examples directory has 266 test functions in 24 files. In every suite summary, 195 Triton tests run and 8,254 are skipped. My own collection counted about 8,451 items per target, of which 195 run and 8,256 skip (test_core.py 8,228, test_annotations.py 26, test_bitcast_ptr.py 2), so my count and the summary differ by 2 skipped items; I did not find which. torch collects 21 tests (the runbook says 20).

## 3. Phase A timing (Canoe phone `5a28e3c2`)

Session `A_bench`: matmul on the A build, variant `base` (patch A's switches off) against `all`, 5 reps, interleaved, 80 runs, all outputs matched, 0 of 16 cases with rep spread over 10 %. The run was interrupted at 77 of 80 by the loss of the orchestrator process and resumed; the last 3 runs (rep 5) came from the same source and settings, but from a rebuilt `libtriton.so` (hash 5e44eb926f57; the hash of the build used for the first 77 runs was not recorded). The full table as written by `summarize.py` is in the appendix.

| Matmul MxKxN | Target | base median ms | all median ms | Speedup | base min-max | all min-max |
|---|---|---:|---:|---:|---|---|
| 179x167x321 | sme | 3706.1 | 3637.4 | 1.019x | 3637.6-3724.7 | 3620.9-3690.1 |
| 179x167x321 | sve | 217.9 | 208.5 | 1.045x | 210.7-228.6 | 204.8-217.5 |
| 256x256x256 | sme | 4721.7 | 4662.5 | 1.013x | 4662.2-4761.4 | 4648.9-4719.0 |
| 256x256x256 | sve | 236.4 | 220.4 | 1.072x | 227.9-245.4 | 216.6-232.2 |
| 512x512x512 | sme | 36478.8 | 36450.0 | 1.001x | 36456.7-36523.9 | 36419.5-36481.6 |
| 512x512x512 | sve | 793.9 | 667.2 | 1.190x | 766.4-806.7 | 653.6-674.3 |
| 1024x1024x1024 | sme | 291635.5 | 292126.4 | 0.998x | 290961.6-294014.1 | 290481.0-300925.6 |
| 1024x1024x1024 | sve | 5667.1 | 4580.2 | 1.237x | 5272.2-5787.9 | 4299.3-4687.6 |

- matmul 179x167x321 sme: 3706.1 ms -> 3637.4 ms (1.02 x)
- matmul 179x167x321 sve: 217.9 ms -> 208.5 ms (1.04 x)
- matmul 256x256x256 sme: 4721.7 ms -> 4662.5 ms (1.01 x)
- matmul 256x256x256 sve: 236.4 ms -> 220.4 ms (1.07 x)
- matmul 512x512x512 sme: 36478.8 ms -> 36450.0 ms (1.00 x)
- matmul 512x512x512 sve: 793.9 ms -> 667.2 ms (1.19 x)
- matmul 1024x1024x1024 sme: 291635.5 ms -> 292126.4 ms (1.00 x)
- matmul 1024x1024x1024 sve: 5667.1 ms -> 4580.2 ms (1.24 x)

The runbook expected a larger gain than qemu's 24 % on sme. On the phone, sme shows no gain at any size (all within 2 %), while sve gains at every size, most at 512³ and 1024³.

sme is much slower than sve on the same matmul (matrix session, main build): 179x167x321: 17.0x, 256x256x256: 19.7x, 512x512x512: 46.4x, 1024x1024x1024: 53.4x (sme time / sve time).

## 4. Phase B tests (combined patch)

### 4.1 qemu, B (`B_qemu/summary.txt`)
```
lit:  Passed : 16 (94.12%)  Unresolved: 1 (5.88%)
triton sme: 195 passed, 8254 skipped, 74 warnings in 558.47s (0:09:18)
torch  sme: 21 passed in 400.15s (0:06:40)
triton sve: 195 passed, 8254 skipped, 74 warnings in 558.62s (0:09:18)
torch  sve: 21 passed in 237.38s (0:03:57)
```
One more lit test passes than in A (16 vs 15), which is PR #14's `arm64-reduction-scheduling.mlir`.

### 4.2 PR #14's own lit tests (`B_pr14_lit.txt`)
```
PASS: TRITON-ARM64 :: Transforms/arm64-reduction-scheduling.mlir (9 of 17)
PASS: TRITON-ARM64 :: Transforms/optimize-triton-shared-softmax.mlir (14 of 17)
```

### 4.3 Reductions in streaming mode, qemu at two SVLs (`ssve_qemu_summary.txt`)
```
svl=128 pr14: triton 10 failed, 33 passed in 88.73s (0:01:28) | torch 10 failed, 3 passed in 34.38s
svl=128 pr14fix: triton 43 passed in 115.73s (0:01:55) | torch 13 passed in 34.83s
svl=512 pr14: triton 10 failed, 33 passed in 87.95s (0:01:27) | torch 10 failed, 3 passed in 34.14s
svl=512 pr14fix: triton 43 passed in 117.80s (0:01:57) | torch 13 passed in 35.74s
```
`pr14` as posted fails 10 Triton + 10 torch at both SVL 128 and SVL 512; `pr14fix` passes everything (43 Triton, 13 torch). This reproduces the runbook's expectation.

### 4.4 Reductions in streaming mode, Canoe phone (`ssve_phone_summary.txt`)
```
phone ssve (pr14fix + SSVE, sme): triton 43 passed in 497.29s (0:08:17) | torch 13 passed in 123.44s (0:02:03)
```
Matches the expected 43 Triton + 13 torch.

## 5. Phase B timing (Canoe phone `5a28e3c2`)

Softmax on the B build, variants switched by environment variables. Two sessions: `B_bench` (5 reps, 200 runs; spread check: 3 of 40 cases with rep spread > 10%) and, because of that spread, `B_bench_reps9` (9 reps, 360 runs; 5 of 40 cases with rep spread > 10%). The 9-rep run is on a rebuilt B (different `libtriton.so` hash); both sessions' full tables are in the appendix. Variants: `main` = PR #14 off (but patch A still present, so not origin/main), `pr14` as posted, `pr14fix` (adds the unmask switch), `reassocfix` (reassoc only, no unroll); `ssve_*` = also run reductions in streaming mode, sme only.

Cases with spread over 10 %:

```
5-rep:
  sme main torchx64x8192: min 93.393 max 111.350 spread 19.2%
  sme reassocfix torchx64x8192: min 93.316 max 104.970 spread 12.5%
  sme ssve_pr14fix torchx64x8192: min 85.603 max 94.233 spread 10.1%
9-rep:
  sme pr14 torchx64x8192: min 96.588 max 107.143 spread 10.9%
  sme ssve_reassocfix torchx64x8192: min 79.027 max 98.459 spread 24.6%
  sme ssve_reassocfix tritonx256x2048: min 72.464 max 80.537 spread 11.1%
  sve main tritonx4x8192: min 8.068 max 9.039 spread 12.0%
  sve reassocfix torchx64x8192: min 84.605 max 102.555 spread 21.2%
```

### 5.1 9-rep session (`B_bench_reps9`)

Median ms per 20 iterations (speedup vs `main` in brackets).

**sme**

| Size | main | pr14 | pr14fix | reassocfix | ssve_pr14fix | ssve_reassocfix |
|---|---:|---:|---:|---:|---:|---:|
| triton 4x8192 | 8.60 (1.000x) | 7.59 (1.133x) | 7.40 (1.161x) | 7.35 (1.170x) | 7.64 (1.126x) | 7.65 (1.124x) |
| triton 64x8192 | 92.96 (1.000x) | 83.27 (1.116x) | 82.85 (1.122x) | 79.67 (1.167x) | 82.98 (1.120x) | 80.20 (1.159x) |
| triton 256x2048 | 91.30 (1.000x) | 80.72 (1.131x) | 80.06 (1.140x) | 80.75 (1.131x) | 75.99 (1.201x) | 76.52 (1.193x) |
| torch 64x8192 | 110.68 (1.000x) | 101.28 (1.093x) | 101.77 (1.088x) | 98.54 (1.123x) | 92.44 (1.197x) | 91.40 (1.211x) |

**sve**

| Size | main | pr14 | pr14fix | reassocfix |
|---|---:|---:|---:|---:|
| triton 4x8192 | 8.60 (1.000x) | 7.58 (1.134x) | 7.38 (1.165x) | 7.35 (1.170x) |
| triton 64x8192 | 89.80 (1.000x) | 83.06 (1.081x) | 81.38 (1.103x) | 81.62 (1.100x) |
| triton 256x2048 | 92.45 (1.000x) | 83.86 (1.102x) | 82.27 (1.124x) | 81.26 (1.138x) |
| torch 64x8192 | 107.37 (1.000x) | 99.84 (1.075x) | 103.57 (1.037x) | 100.39 (1.070x) |

### 5.2 5-rep session (`B_bench`)

Median ms per 20 iterations (speedup vs `main` in brackets).

**sme**

| Size | main | pr14 | pr14fix | reassocfix | ssve_pr14fix | ssve_reassocfix |
|---|---:|---:|---:|---:|---:|---:|
| triton 4x8192 | 8.59 (1.000x) | 7.63 (1.126x) | 7.36 (1.167x) | 7.35 (1.169x) | 7.61 (1.129x) | 7.68 (1.119x) |
| triton 64x8192 | 93.54 (1.000x) | 84.67 (1.105x) | 81.05 (1.154x) | 80.58 (1.161x) | 84.13 (1.112x) | 82.05 (1.140x) |
| triton 256x2048 | 89.27 (1.000x) | 83.98 (1.063x) | 81.68 (1.093x) | 79.34 (1.125x) | 74.51 (1.198x) | 74.67 (1.196x) |
| torch 64x8192 | 109.44 (1.000x) | 100.99 (1.084x) | 102.08 (1.072x) | 97.65 (1.121x) | 92.07 (1.189x) | 94.17 (1.162x) |

**sve**

| Size | main | pr14 | pr14fix | reassocfix |
|---|---:|---:|---:|---:|
| triton 4x8192 | 8.59 (1.000x) | 7.57 (1.135x) | 7.35 (1.169x) | 7.36 (1.167x) |
| triton 64x8192 | 87.25 (1.000x) | 80.08 (1.090x) | 83.76 (1.042x) | 83.53 (1.045x) |
| triton 256x2048 | 94.52 (1.000x) | 86.23 (1.096x) | 79.78 (1.185x) | 79.52 (1.189x) |
| torch 64x8192 | 109.36 (1.000x) | 100.92 (1.084x) | 100.81 (1.085x) | 97.08 (1.126x) |

### 5.3 The three questions of the runbook

Each cell: 5-rep session / 9-rep session. Each ratio is median(first-named variant) / median(second-named variant), so above 1 means the **second**-named variant is faster.

**Q1. Does PR #14 as posted beat `main` on the phone?** (`main` median / `pr14` median)

| Target | tritonx4x8192 | tritonx64x8192 | tritonx256x2048 | torchx64x8192 |
|---|---|---|---|---|
| sme | 1.126 / 1.133 | 1.105 / 1.116 | 1.063 / 1.131 | 1.084 / 1.093 |
| sve | 1.135 / 1.134 | 1.090 / 1.081 | 1.096 / 1.102 | 1.084 / 1.075 |

Yes on the Triton sizes (about 1.08x to 1.13x in the 9-rep session); torch 64x8192 about 1.08x to 1.09x but this case is noisy (19 % spread on `main` in the 5-rep session).

**Q2. Does the fix beat PR #14? Is `pr14fix` (unroll) better than `reassocfix` (no unroll)?**

Columns: median(pr14) / median(pr14fix) (above 1 = `pr14fix` faster) and median(pr14fix) / median(reassocfix) (above 1 = `reassocfix`, no unroll, faster).

| Target | Size | pr14 / pr14fix (5r / 9r) | pr14fix / reassocfix (5r / 9r) |
|---|---|---|---|
| sme | tritonx4x8192 | 1.037 / 1.025 | 1.002 / 1.008 |
| sme | tritonx64x8192 | 1.045 / 1.005 | 1.006 / 1.040 |
| sme | tritonx256x2048 | 1.028 / 1.008 | 1.029 / 0.991 |
| sme | torchx64x8192 | 0.989 / 0.995 | 1.045 / 1.033 |
| sve | tritonx4x8192 | 1.030 / 1.028 | 0.999 / 1.004 |
| sve | tritonx64x8192 | 0.956 / 1.021 | 1.003 / 0.997 |
| sve | tritonx256x2048 | 1.081 / 1.019 | 1.003 / 1.012 |
| sve | torchx64x8192 | 1.001 / 0.964 | 1.038 / 1.032 |

`pr14fix` vs `pr14`: between 0.956 and 1.081 across the 16 cells (9-rep session: 0.964 to 1.028), so `pr14fix` is at best a few percent faster and sometimes slower; this is the size of the rep spread. `reassocfix` vs `pr14fix`: between 0.991 and 1.045; `reassocfix` is never meaningfully slower, and is consistently about 3 to 4 % faster on torch 64x8192 (both sessions, both targets) and level on the Triton sve cases. So in these data the 4-way unroll adds nothing measurable once `reassoc` reaches the assembly, which is the runbook's "reassocfix ≈ pr14fix" outcome; differences of 1 to 3 % are inside the rep spread and I did not run enough reps to separate them.

**Q3. Does SSVE (`ssve_*`) beat the same variant without it on sme, at this phone's SVL 512?** (non-SSVE median / SSVE median; > 1 means SSVE is faster)

| Pair | tritonx4x8192 | tritonx64x8192 | tritonx256x2048 | torchx64x8192 |
|---|---|---|---|---|
| pr14fix vs ssve_pr14fix | 0.967 / 0.969 | 0.963 / 0.998 | 1.096 / 1.054 | 1.109 / 1.101 |
| reassocfix vs ssve_reassocfix | 0.958 / 0.961 | 0.982 / 0.993 | 1.063 / 1.055 | 1.037 / 1.078 |

(cells: 5-rep / 9-rep). SSVE is faster on torch 64x8192 (1.04x to 1.11x) and Triton 256x2048 (1.05x to 1.10x), about 3 to 4 % slower on Triton 4x8192 (0.96x to 0.97x), and neutral to 4 % slower on Triton 64x8192 (0.96x to 1.00x). So the answer is mixed: SSVE helps the larger/torch cases and hurts the smallest Triton one. The torch 64x8192 case is also the noisiest one (spread 10 to 25 % in some variants), so its gain is the least certain. The runbook's qemu context (SSVE 4 % worse at SVL 128, 15 % fewer instructions at SVL 512) does not predict these phone numbers at SVL 512 directly: I did not measure instruction counts for SSVE on the phone.

## 6. Optional runs

### 6.1 True main as a control (`main_bench`, Canoe)

origin/main (no patch), matmul 512x512x512, variant `all` (= main), 5 reps, 0 of 2 cases with spread over 10 %. Compared with `A_bench` `base` (a different session and build):

| Target | main (this control) ms | A bench `base` ms | A bench `all` ms |
|---|---:|---:|---:|
| sme | 36532.6 | 36478.8 | 36450.0 |
| sve | 761.7 | 793.9 | 667.2 |

sme: main 36532.6 ms vs `base` 36478.8 ms (0.15 % apart), so `base` behaves as main, as claimed. sve: main 761.7 ms vs `base` 793.9 ms: the control is 4.1 % faster than `base`. This is a different session and a different build, and I did not investigate the sve difference further; the matrix (section 6.3) is the paired comparison against main. The IR check agrees on the code: `base` and the main artifact have identical LLVM IR at 512³ for both targets.

### 6.2 llvm-mca view of the reduction loop (`B_mca`, qemu for correctness, triton 4x8192 sve)
```
== main (triton 4x8192 sve ssve=0): triton matched: True
loop at .LBB0_8: 20 insns, 1x fmaxnmv -> 4.1 cycles/iteration
loop at .LBB0_12: 17 insns, 2x fadda -> 12.1 cycles/iteration
== pr14 (triton 4x8192 sve ssve=0): triton matched: True
loop at .LBB0_8: 20 insns, 1x fmaxnmv -> 4.1 cycles/iteration
loop at .LBB0_12: 69 insns, 8x fadda -> 40.1 cycles/iteration
== pr14fix (triton 4x8192 sve ssve=0): triton matched: True
loop at .LBB0_8: 20 insns, 1x fmaxnmv -> 4.1 cycles/iteration
loop at .LBB0_12: 61 insns, 4x fadd, 4x faddv -> 12.6 cycles/iteration
== reassocfix (triton 4x8192 sve ssve=0): triton matched: True
loop at .LBB0_8: 20 insns, 1x fmaxnmv -> 4.1 cycles/iteration
loop at .LBB0_12: 15 insns, 1x fadd, 1x faddv -> 3.1 cycles/iteration
```
main (12.1) and reassocfix (3.1) match the runbook's expected cycles/iteration. `pr14` (40.1 vs expected 10.0) and `pr14fix` (12.6 vs expected 3.2) do **not**: this run shows 8 `fadda` in `pr14`, and `pr14fix` still ends with the larger loop. I did not find the cause (see section 7). The mca figure for `pr14` (40.1, worse than `main`) also does not agree with the phone, where `pr14` is 1.08x to 1.13x faster than `main` (section 5.3); the phone timings are the measurement, mca is only a model.

### 6.3 Three-build matrix on Canoe (`matrix/`)

One interleaved session, 24 configs x labels main / A / B x 5 reps = 360 runs, all rc=0, all 360 outputs matched their reference (`verify.txt`: 24 of 24 configs all checks ok). Each run uses the executable compiled once per label; each timing is the executable's own 20-iteration benchmark. Speedup = median(main) / median(label). The order inside each rep is fixed: main, A, B.

| config | main median [min-max] (n) | A median [min-max] (n) | B median [min-max] (n) | speedup A vs main | speedup B vs main | per-rep ratios A | per-rep ratios B |
|---|---|---|---|---|---|---|---|
| sme_mm_1024x1024x1024 | 291321.10 [290989.09-291524.91] (5) | 297075.17 [291153.68-297235.77] (5) | 304738.97 [297098.47-305297.91] (5) | **0.981x** | **0.956x** | 0.999 / 0.985 / 0.981 / 0.980 / 0.981 | 0.979 / 0.960 / 0.957 / 0.955 / 0.954 |
| sme_mm_179x167x321 | 3690.56 [3639.51-3729.28] (5) | 3693.82 [3640.13-3711.49] (5) | 3671.23 [3639.03-3711.03] (5) | **0.999x** | **1.005x** | 0.993 / 1.016 / 1.010 / 0.985 / 0.994 | 0.987 / 1.000 / 1.025 / 0.995 / 1.005 |
| sme_mm_256x256x256 | 4722.21 [4703.02-4748.97] (5) | 4721.67 [4704.65-4727.10] (5) | 4715.65 [4651.04-4738.51] (5) | **1.000x** | **1.001x** | 0.999 / 0.996 / 1.003 / 1.002 / 1.005 | 1.001 / 1.011 / 1.002 / 0.998 / 1.002 |
| sme_mm_512x512x512 | 36533.35 [36450.04-36559.87] (5) | 36492.70 [36476.01-36516.78] (5) | 36476.60 [36405.90-36482.29] (5) | **1.001x** | **1.002x** | 1.001 / 1.000 / 1.001 / 0.999 / 1.002 | 1.002 / 1.004 / 1.001 / 0.999 / 1.002 |
| sme_rd_torch_256x781 | 11.27 [10.26-11.34] (5) | 11.26 [10.50-11.31] (5) | 5.63 [5.34-5.80] (5) | **1.000x** | **2.000x** | 1.000 / 0.978 / 0.998 / 1.069 / 0.925 | 2.000 / 1.821 / 2.112 / 1.955 / 1.921 |
| sme_rd_torch_64x8192 | 26.04 [25.31-27.66] (5) | 25.94 [25.46-28.42] (5) | 13.39 [12.99-14.42] (5) | **1.004x** | **1.944x** | 0.976 / 0.934 / 1.066 / 0.921 / 1.023 | 1.883 / 1.906 / 2.123 / 2.014 / 1.806 |
| sme_rd_triton_256x781 | 16.95 [16.06-17.72] (5) | 16.72 [16.00-17.51] (5) | 10.05 [9.11-10.08] (5) | **1.014x** | **1.687x** | 1.041 / 0.944 / 1.068 / 0.963 / 1.059 | 1.911 / 1.593 / 1.764 / 1.679 / 1.816 |
| sme_rd_triton_4x8192 | 1.89 [1.87-1.90] (5) | 1.87 [1.69-1.90] (5) | 0.89 [0.80-0.89] (5) | **1.012x** | **2.126x** | 1.001 / 1.013 / 1.107 / 0.997 / 0.999 | 2.099 / 2.128 / 2.353 / 2.126 / 2.128 |
| sme_sm_torch_64x8192 | 104.02 [101.09-111.25] (5) | 107.78 [92.87-109.44] (5) | 102.28 [97.59-106.03] (5) | **0.965x** | **1.017x** | 0.936 / 0.938 / 0.966 / 1.003 / 1.198 | 1.022 / 0.988 / 1.066 / 1.035 / 1.067 |
| sme_sm_triton_256x2048 | 89.13 [85.84-92.82] (5) | 88.75 [87.44-90.39] (5) | 83.77 [80.07-85.96] (5) | **1.004x** | **1.064x** | 1.046 / 0.988 / 1.012 / 0.976 / 0.988 | 1.080 / 1.061 / 1.073 / 1.072 / 1.064 |
| sme_sm_triton_4x8192 | 8.59 [8.39-8.61] (5) | 8.60 [7.88-8.62] (5) | 7.57 [7.57-7.58] (5) | **0.999x** | **1.134x** | 0.997 / 1.000 / 1.049 / 1.093 / 0.996 | 1.133 / 1.136 / 1.108 / 1.136 / 1.134 |
| sme_sm_triton_64x8192 | 90.62 [89.55-93.38] (5) | 92.05 [88.56-93.39] (5) | 83.23 [79.02-85.84] (5) | **0.984x** | **1.089x** | 0.995 / 1.023 / 1.011 / 0.963 / 0.990 | 1.080 / 1.097 / 1.088 / 1.080 / 1.133 |
| sve_mm_1024x1024x1024 | 5456.25 [5438.00-5458.40] (5) | 4468.65 [4447.38-4485.07] (5) | 4473.40 [4432.54-4488.54] (5) | **1.221x** | **1.220x** | 1.223 / 1.217 / 1.219 / 1.219 / 1.223 | 1.227 / 1.220 / 1.220 / 1.216 / 1.220 |
| sve_mm_179x167x321 | 217.73 [214.37-222.88] (5) | 213.16 [204.75-213.77] (5) | 206.61 [198.33-211.34] (5) | **1.021x** | **1.054x** | 1.047 / 1.007 / 1.034 / 1.045 / 1.043 | 1.081 / 1.016 / 1.084 / 1.058 / 1.054 |
| sve_mm_256x256x256 | 239.35 [230.28-240.66] (5) | 220.94 [211.88-230.30] (5) | 221.60 [209.24-229.96] (5) | **1.083x** | **1.080x** | 1.000 / 1.065 / 1.130 / 1.082 / 1.089 | 1.101 / 1.054 / 1.041 / 1.052 / 1.086 |
| sve_mm_512x512x512 | 787.08 [772.06-797.20] (5) | 669.10 [664.87-678.87] (5) | 672.97 [668.09-682.90] (5) | **1.176x** | **1.170x** | 1.177 / 1.176 / 1.156 / 1.169 / 1.174 | 1.193 / 1.170 / 1.153 / 1.163 / 1.143 |
| sve_rd_torch_256x781 | 11.24 [10.83-11.75] (5) | 11.26 [10.77-11.29] (5) | 5.64 [5.21-5.66] (5) | **0.998x** | **1.993x** | 1.031 / 1.001 / 1.009 / 1.041 / 0.960 | 1.969 / 1.999 / 1.987 / 2.236 / 2.078 |
| sve_rd_torch_64x8192 | 25.93 [25.79-27.18] (5) | 27.27 [25.08-28.15] (5) | 13.78 [12.80-14.27] (5) | **0.951x** | **1.882x** | 0.916 / 0.951 / 1.083 / 0.923 / 1.054 | 1.808 / 1.882 / 2.112 / 2.026 / 1.885 |
| sve_rd_triton_256x781 | 16.84 [16.59-17.29] (5) | 16.16 [15.95-16.93] (5) | 9.59 [9.34-10.10] (5) | **1.042x** | **1.757x** | 1.035 / 1.053 / 1.042 / 0.998 / 1.029 | 1.730 / 1.799 / 1.773 / 1.674 / 1.727 |
| sve_rd_triton_4x8192 | 1.87 [1.87-1.95] (5) | 1.90 [1.85-2.00] (5) | 0.89 [0.89-0.89] (5) | **0.987x** | **2.099x** | 0.960 / 1.041 / 1.001 / 1.009 / 0.935 | 2.098 / 2.186 / 2.129 / 2.099 / 2.098 |
| sve_sm_torch_64x8192 | 105.36 [96.64-107.22] (5) | 108.36 [103.65-110.03] (5) | 102.28 [96.03-110.19] (5) | **0.972x** | **1.030x** | 0.994 / 0.892 / 0.974 / 1.027 / 0.968 | 1.089 / 0.947 / 0.973 / 1.007 / 1.030 |
| sve_sm_triton_256x2048 | 89.84 [85.82-92.48] (5) | 86.64 [82.36-92.84] (5) | 79.03 [78.68-85.56] (5) | **1.037x** | **1.137x** | 1.062 / 1.003 / 0.996 / 1.005 / 1.058 | 1.107 / 1.078 / 1.175 / 1.050 / 1.165 |
| sve_sm_triton_4x8192 | 8.47 [8.07-8.61] (5) | 8.28 [8.09-8.59] (5) | 7.57 [7.19-7.60] (5) | **1.022x** | **1.118x** | 1.046 / 1.026 / 1.003 / 0.940 / 1.038 | 1.114 / 1.101 / 1.134 / 1.123 / 1.135 |
| sve_sm_triton_64x8192 | 88.12 [79.74-92.07] (5) | 88.98 [85.60-92.39] (5) | 83.66 [80.94-84.77] (5) | **0.990x** | **1.053x** | 0.869 / 1.073 / 1.022 / 0.954 / 1.008 | 0.961 / 1.138 / 1.044 / 1.040 / 1.072 |

LLVM IR identity, main vs label (`ir_identity.txt`), number of configs per family:

| Target | Family | A vs main | B vs main |
|---|---|---|---|
| sme | matmul | 4 differs | 4 differs |
| sme | softmax |  |  |
| sme | row sum |  |  |
| sve | matmul | 4 differs | 4 differs |
| sve | softmax |  |  |
| sve | row sum |  |  |

### 6.4 HAWI (`hawi/`), A vs main only

Device `e8ab1b5` on `hyd-lablnx369`, SVL 512. A trimmed set, as requested: matmul at four sizes on sme and sve, plus Triton softmax 4x8192 and row sum 4x8192 on sme. 3 reps, 60 runs, labels main and A, all rc=0, all 60 outputs matched (`verify.txt`). B was not run on HAWI.

Correctness (`hawi/pytest_A/summary.txt`, `hawi/pytest_main/summary.txt`): transpose, flash attention, matmul and mm tests (Triton) and transpose, add+matmul+softmax and concat+linear (torch):
```
A triton sme: 15 passed in 267.42s (0:04:27)
A torch sme: 4 passed in 475.03s (0:07:55)
A triton sve: 15 passed in 206.44s (0:03:26)
A torch sve: 4 passed in 481.84s (0:08:01)
main triton sme: 15 passed in 262.00s (0:04:22)
main torch sme: 4 passed in 477.28s (0:07:57)
main triton sve: 15 passed in 210.15s (0:03:30)
main torch sve: 4 passed in 466.28s (0:07:46)
```

| config | main median [min-max] (n) | A median [min-max] (n) | speedup A vs main | per-rep ratios A |
|---|---|---|---|---|
| sme_mm_1024x1024x1024 | 297009.87 [296450.28-297088.07] (3) | 296622.56 [295901.31-311640.11] (3) | **1.001x** | 1.002 / 1.002 / 0.953 |
| sme_mm_179x167x321 | 3650.30 [3614.83-3660.64] (3) | 3728.96 [3620.42-3759.55] (3) | **0.979x** | 0.998 / 0.971 / 0.982 |
| sme_mm_256x256x256 | 4654.12 [4638.00-4710.85] (3) | 4794.30 [4700.33-4827.23] (3) | **0.971x** | 0.964 / 0.967 / 1.002 |
| sme_mm_512x512x512 | 36720.68 [36627.05-37458.30] (3) | 36674.02 [36585.08-36949.16] (3) | **1.001x** | 1.001 / 1.014 / 1.001 |
| sme_rd_triton_4x8192 | 1.16 [1.16-1.68] (3) | 1.65 [1.16-4.15] (3) | **0.703x** | 0.702 / 1.444 / 0.280 |
| sme_sm_triton_4x8192 | 5.45 [5.32-7.67] (3) | 6.77 [6.72-7.90] (3) | **0.804x** | 0.785 / 1.141 / 0.689 |
| sve_mm_1024x1024x1024 | 4532.07 [4474.29-4556.49] (3) | 3798.99 [3714.05-3803.48] (3) | **1.193x** | 1.205 / 1.192 / 1.199 |
| sve_mm_179x167x321 | 160.41 [152.29-177.95] (3) | 147.19 [136.42-167.72] (3) | **1.090x** | 0.908 / 1.304 / 1.090 |
| sve_mm_256x256x256 | 207.28 [165.24-207.29] (3) | 153.02 [114.62-180.49] (3) | **1.355x** | 1.148 / 1.355 / 1.442 |
| sve_mm_512x512x512 | 687.68 [625.74-701.73] (3) | 558.62 [547.59-561.05] (3) | **1.231x** | 1.120 / 1.251 / 1.256 |

The softmax and row-sum LLVM IR is identical in main and A on HAWI, so their ratios below 1 are noise on 1-7 ms runs with 3 reps. The matmul IR differs (as expected).

### 6.5 A vs main across all sessions (not mixed: each cell comes from its own session)

| Config | Canoe A bench (base to all, same source, switches only) | Canoe matrix (main to A, separate builds) | HAWI (main to A, separate builds) |
|---|---|---|---|
| sve matmul 179x167x321 | 1.045x | 1.021x | 1.090x |
| sve matmul 256x256x256 | 1.072x | 1.083x | 1.355x |
| sve matmul 512x512x512 | 1.190x | 1.176x | 1.231x |
| sve matmul 1024x1024x1024 | 1.237x | 1.221x | 1.193x |
| sme matmul 179x167x321 | 1.019x | 0.999x | 0.979x |
| sme matmul 256x256x256 | 1.013x | 1.000x | 0.971x |
| sme matmul 512x512x512 | 1.001x | 1.001x | 1.001x |
| sme matmul 1024x1024x1024 | 0.998x | 0.981x | 1.001x |

## 7. Anomalies and caveats

- **Lost orchestrators.** The phone orchestrator and the waiting matrix script were killed around 17:00 on 23 Nov (cause unknown; no error in any log). The Canoe A matmul bench had 77 of 80 runs and was resumed. I reported progress at 17:13 without checking the process was alive. After that I wrote a resumable runner (`resume_all.sh`, checkpoint files in `done/`, run-level resume in `device_bench.sh` and `matrix_sweep.sh`) and everything after that ran through it.
- **First attempt at the B 9-rep rerun did not run** (my script passed an empty argument to the bench, rc=127). It was fixed and rerun; `B_bench_reps9` is from the second attempt.
- **Rep spread rule.** `A_bench`: 0 of 16 cases over 10 %. `main_bench`: 0 of 2. `B_bench`: 3 of 40, so it was rerun with 9 reps; the 9-rep run still has 5 of 40 over 10 % (torch 64x8192 on sme and sve, Triton 256x2048 and 4x8192 in one variant each). Those cases are noisy; I did not rerun further. The matrix and HAWI sweeps are outside the runbook's rule; their min-max are in the tables.
- **sme 1024³ on Canoe (matrix).** A is 0.981x and B is 0.956x relative to main. A and B have identical LLVM IR for matmul (0 changed lines), so the extra slowdown of B is not caused by its changes. The medians rise in run order inside each rep (main 291.3 s, A 297.1 s, B 304.7 s). A drift from phone warm-up over runs of about 5 minutes each is plausible, but I did not test it by reversing the order. On HAWI the same config shows 1.001x. The `A_bench` session (same source, switches only) shows 0.998x.
- **Different builds of the same label.** The compiled `libtriton.so` of A was rebuilt several times and is not bit-identical across rebuilds (hash 5e44eb926f57 in the A bench, 5af3c6bb3f48 in the matrix). B has hashes b9aaa6a368ab (first B bench and matrix) and 5c8b14338980 (the 9-rep rerun). Results from different builds of the same label were never combined.
- **`main` variant of the softmax bench is not origin/main**: it is PR #14 switched off inside build B (patch A still present). The three-build matrix is the comparison against real origin/main.
- **llvm-mca cycles do not match the runbook for `pr14` and `pr14fix`** (40.1 and 12.6 vs expected 10.0 and 3.2); `main` (12.1) and `reassocfix` (3.1) match. Cause not found. It does not affect the device timings above, which are measured on the phone.
- **Unresolved lit item** is the scratch file `a.mlir` in every lit run, including main.
- **Test counts.** The suite summary and my own collection differ by 2 skipped items (8,254 vs 8,256); torch has 21 tests, not 20 as the runbook says.
- **Patch A also carries the `TRITON_ARM64_SSVE_REDUCTIONS` option** (default off). Whether it belongs in the PR was not decided here.
- **HAWI:** only A vs main, a trimmed config set, 3 reps (so noisy on the smallest runs), and the serial `e8ab1b5` was assumed to be the intended device.
- **adb.** My last check on `hyd-lablnx294` used an adb client (v39) that restarted the host's adb server (v41); the three phones were listed again immediately and no run was active. Nothing else was affected.
- **Not run:** nothing from the runbook was skipped. `status_before.txt` was never recorded, so the final `git status` could not be compared against it.

## Appendix: raw result tables

### A bench (`A_bench/RESULTS.md`)

### Results (/local/mnt/workspace/adittriv/triton-mlir/sme_ext_results/A_bench)

speedup = baseline median / variant median (> 1 = faster). ms = whole grid x ITERS iterations, measured inside the exe (benchmark.log).

#### matmul 1024x1024x1024 on sme (device)

| variant | median ms | min ms | reps ok/total | speedup vs base | all match | smstart in asm |
|---|---:|---:|---:|---:|---|---:|
| base | 291635.550 | 290961.615 | 5/5 | 1.00x | yes | 17 |
| all | 292126.387 | 290481.017 | 5/5 | 1.00x | yes | 4 |

#### matmul 179x167x321 on sme (device)

| variant | median ms | min ms | reps ok/total | speedup vs base | all match | smstart in asm |
|---|---:|---:|---:|---:|---|---:|
| base | 3706.143 | 3637.583 | 5/5 | 1.00x | yes | 17 |
| all | 3637.405 | 3620.853 | 5/5 | 1.02x | yes | 4 |

#### matmul 256x256x256 on sme (device)

| variant | median ms | min ms | reps ok/total | speedup vs base | all match | smstart in asm |
|---|---:|---:|---:|---:|---|---:|
| base | 4721.746 | 4662.157 | 5/5 | 1.00x | yes | 17 |
| all | 4662.523 | 4648.852 | 5/5 | 1.01x | yes | 4 |

#### matmul 512x512x512 on sme (device)

| variant | median ms | min ms | reps ok/total | speedup vs base | all match | smstart in asm |
|---|---:|---:|---:|---:|---|---:|
| base | 36478.752 | 36456.669 | 5/5 | 1.00x | yes | 17 |
| all | 36449.962 | 36419.502 | 5/5 | 1.00x | yes | 4 |

#### matmul 1024x1024x1024 on sve (device)

| variant | median ms | min ms | reps ok/total | speedup vs base | all match | smstart in asm |
|---|---:|---:|---:|---:|---|---:|
| base | 5667.069 | 5272.239 | 5/5 | 1.00x | yes | 0 |
| all | 4580.216 | 4299.281 | 5/5 | 1.24x | yes | 0 |

#### matmul 179x167x321 on sve (device)

| variant | median ms | min ms | reps ok/total | speedup vs base | all match | smstart in asm |
|---|---:|---:|---:|---:|---|---:|
| base | 217.889 | 210.674 | 5/5 | 1.00x | yes | 0 |
| all | 208.510 | 204.825 | 5/5 | 1.04x | yes | 0 |

#### matmul 256x256x256 on sve (device)

| variant | median ms | min ms | reps ok/total | speedup vs base | all match | smstart in asm |
|---|---:|---:|---:|---:|---|---:|
| base | 236.369 | 227.861 | 5/5 | 1.00x | yes | 0 |
| all | 220.435 | 216.569 | 5/5 | 1.07x | yes | 0 |

#### matmul 512x512x512 on sve (device)

| variant | median ms | min ms | reps ok/total | speedup vs base | all match | smstart in asm |
|---|---:|---:|---:|---:|---|---:|
| base | 793.940 | 766.445 | 5/5 | 1.00x | yes | 0 |
| all | 667.243 | 653.555 | 5/5 | 1.19x | yes | 0 |

### main control (`main_bench/RESULTS.md`)

### Results (/local/mnt/workspace/adittriv/triton-mlir/sme_ext_results/main_bench)

speedup = baseline median / variant median (> 1 = faster). ms = whole grid x ITERS iterations, measured inside the exe (benchmark.log).

#### matmul 512x512x512 on sme (device)

| variant | median ms | min ms | reps ok/total | speedup vs base | all match | smstart in asm |
|---|---:|---:|---:|---:|---|---:|
| all | 36532.579 | 36525.794 | 5/5 | NA | yes | 17 |

#### matmul 512x512x512 on sve (device)

| variant | median ms | min ms | reps ok/total | speedup vs base | all match | smstart in asm |
|---|---:|---:|---:|---:|---|---:|
| all | 761.653 | 758.224 | 5/5 | NA | yes | 0 |

### B bench, 5 reps (`B_bench/RESULTS.md`)

### Results (/local/mnt/workspace/adittriv/triton-mlir/sme_ext_results/B_bench)

speedup = baseline median / variant median (> 1 = faster). ms = whole grid x ITERS iterations, measured inside the exe (benchmark.log).

#### softmax torchx64x8192 on sme (device)

| variant | median ms | min ms | reps ok/total | speedup vs main | all match | smstart in asm |
|---|---:|---:|---:|---:|---|---:|
| main | 109.440 | 93.393 | 5/5 | 1.00x | yes | 0 |
| pr14 | 100.986 | 95.729 | 5/5 | 1.08x | yes | 0 |
| pr14fix | 102.083 | 101.108 | 5/5 | 1.07x | yes | 0 |
| reassocfix | 97.654 | 93.316 | 5/5 | 1.12x | yes | 0 |
| ssve_pr14fix | 92.070 | 85.603 | 5/5 | 1.19x | yes | 0 |
| ssve_reassocfix | 94.169 | 92.910 | 5/5 | 1.16x | yes | 0 |

#### softmax tritonx256x2048 on sme (device)

| variant | median ms | min ms | reps ok/total | speedup vs main | all match | smstart in asm |
|---|---:|---:|---:|---:|---|---:|
| main | 89.271 | 87.649 | 5/5 | 1.00x | yes | 0 |
| pr14 | 83.981 | 79.194 | 5/5 | 1.06x | yes | 0 |
| pr14fix | 81.677 | 78.505 | 5/5 | 1.09x | yes | 0 |
| reassocfix | 79.342 | 78.788 | 5/5 | 1.13x | yes | 0 |
| ssve_pr14fix | 74.512 | 73.065 | 5/5 | 1.20x | yes | 2 |
| ssve_reassocfix | 74.665 | 74.137 | 5/5 | 1.20x | yes | 2 |

#### softmax tritonx4x8192 on sme (device)

| variant | median ms | min ms | reps ok/total | speedup vs main | all match | smstart in asm |
|---|---:|---:|---:|---:|---|---:|
| main | 8.592 | 8.043 | 5/5 | 1.00x | yes | 0 |
| pr14 | 7.632 | 7.199 | 5/5 | 1.13x | yes | 0 |
| pr14fix | 7.362 | 7.044 | 5/5 | 1.17x | yes | 0 |
| reassocfix | 7.350 | 7.307 | 5/5 | 1.17x | yes | 0 |
| ssve_pr14fix | 7.610 | 7.248 | 5/5 | 1.13x | yes | 2 |
| ssve_reassocfix | 7.676 | 7.608 | 5/5 | 1.12x | yes | 2 |

#### softmax tritonx64x8192 on sme (device)

| variant | median ms | min ms | reps ok/total | speedup vs main | all match | smstart in asm |
|---|---:|---:|---:|---:|---|---:|
| main | 93.543 | 90.557 | 5/5 | 1.00x | yes | 0 |
| pr14 | 84.674 | 81.158 | 5/5 | 1.10x | yes | 0 |
| pr14fix | 81.045 | 79.166 | 5/5 | 1.15x | yes | 0 |
| reassocfix | 80.582 | 78.839 | 5/5 | 1.16x | yes | 0 |
| ssve_pr14fix | 84.132 | 82.780 | 5/5 | 1.11x | yes | 2 |
| ssve_reassocfix | 82.048 | 81.103 | 5/5 | 1.14x | yes | 2 |

#### softmax torchx64x8192 on sve (device)

| variant | median ms | min ms | reps ok/total | speedup vs main | all match | smstart in asm |
|---|---:|---:|---:|---:|---|---:|
| main | 109.358 | 106.864 | 5/5 | 1.00x | yes | 0 |
| pr14 | 100.923 | 99.600 | 5/5 | 1.08x | yes | 0 |
| pr14fix | 100.812 | 97.152 | 5/5 | 1.08x | yes | 0 |
| reassocfix | 97.081 | 95.841 | 5/5 | 1.13x | yes | 0 |

#### softmax tritonx256x2048 on sve (device)

| variant | median ms | min ms | reps ok/total | speedup vs main | all match | smstart in asm |
|---|---:|---:|---:|---:|---|---:|
| main | 94.520 | 91.321 | 5/5 | 1.00x | yes | 0 |
| pr14 | 86.234 | 82.500 | 5/5 | 1.10x | yes | 0 |
| pr14fix | 79.782 | 77.972 | 5/5 | 1.18x | yes | 0 |
| reassocfix | 79.515 | 77.255 | 5/5 | 1.19x | yes | 0 |

#### softmax tritonx4x8192 on sve (device)

| variant | median ms | min ms | reps ok/total | speedup vs main | all match | smstart in asm |
|---|---:|---:|---:|---:|---|---:|
| main | 8.591 | 8.350 | 5/5 | 1.00x | yes | 0 |
| pr14 | 7.570 | 7.331 | 5/5 | 1.13x | yes | 0 |
| pr14fix | 7.350 | 7.328 | 5/5 | 1.17x | yes | 0 |
| reassocfix | 7.361 | 7.304 | 5/5 | 1.17x | yes | 0 |

#### softmax tritonx64x8192 on sve (device)

| variant | median ms | min ms | reps ok/total | speedup vs main | all match | smstart in asm |
|---|---:|---:|---:|---:|---|---:|
| main | 87.248 | 87.205 | 5/5 | 1.00x | yes | 0 |
| pr14 | 80.075 | 79.301 | 5/5 | 1.09x | yes | 0 |
| pr14fix | 83.761 | 78.441 | 5/5 | 1.04x | yes | 0 |
| reassocfix | 83.529 | 78.069 | 5/5 | 1.04x | yes | 0 |

### B bench, 9 reps (`B_bench_reps9/RESULTS.md`)

### Results (/local/mnt/workspace/adittriv/triton-mlir/sme_ext_results/B_bench_reps9)

speedup = baseline median / variant median (> 1 = faster). ms = whole grid x ITERS iterations, measured inside the exe (benchmark.log).

#### softmax torchx64x8192 on sme (device)

| variant | median ms | min ms | reps ok/total | speedup vs main | all match | smstart in asm |
|---|---:|---:|---:|---:|---|---:|
| main | 110.679 | 105.189 | 9/9 | 1.00x | yes | 0 |
| pr14 | 101.281 | 96.588 | 9/9 | 1.09x | yes | 0 |
| pr14fix | 101.770 | 95.006 | 9/9 | 1.09x | yes | 0 |
| reassocfix | 98.538 | 96.019 | 9/9 | 1.12x | yes | 0 |
| ssve_pr14fix | 92.443 | 89.924 | 9/9 | 1.20x | yes | 0 |
| ssve_reassocfix | 91.400 | 79.027 | 9/9 | 1.21x | yes | 0 |

#### softmax tritonx256x2048 on sme (device)

| variant | median ms | min ms | reps ok/total | speedup vs main | all match | smstart in asm |
|---|---:|---:|---:|---:|---|---:|
| main | 91.296 | 88.349 | 9/9 | 1.00x | yes | 0 |
| pr14 | 80.724 | 79.196 | 9/9 | 1.13x | yes | 0 |
| pr14fix | 80.060 | 78.691 | 9/9 | 1.14x | yes | 0 |
| reassocfix | 80.747 | 78.365 | 9/9 | 1.13x | yes | 0 |
| ssve_pr14fix | 75.986 | 73.492 | 9/9 | 1.20x | yes | 2 |
| ssve_reassocfix | 76.516 | 72.464 | 9/9 | 1.19x | yes | 2 |

#### softmax tritonx4x8192 on sme (device)

| variant | median ms | min ms | reps ok/total | speedup vs main | all match | smstart in asm |
|---|---:|---:|---:|---:|---|---:|
| main | 8.597 | 8.072 | 9/9 | 1.00x | yes | 0 |
| pr14 | 7.586 | 7.330 | 9/9 | 1.13x | yes | 0 |
| pr14fix | 7.404 | 6.937 | 9/9 | 1.16x | yes | 0 |
| reassocfix | 7.348 | 7.095 | 9/9 | 1.17x | yes | 0 |
| ssve_pr14fix | 7.638 | 7.465 | 9/9 | 1.13x | yes | 2 |
| ssve_reassocfix | 7.646 | 7.268 | 9/9 | 1.12x | yes | 2 |

#### softmax tritonx64x8192 on sme (device)

| variant | median ms | min ms | reps ok/total | speedup vs main | all match | smstart in asm |
|---|---:|---:|---:|---:|---|---:|
| main | 92.959 | 86.746 | 9/9 | 1.00x | yes | 0 |
| pr14 | 83.267 | 79.703 | 9/9 | 1.12x | yes | 0 |
| pr14fix | 82.853 | 77.437 | 9/9 | 1.12x | yes | 0 |
| reassocfix | 79.675 | 78.322 | 9/9 | 1.17x | yes | 0 |
| ssve_pr14fix | 82.981 | 78.595 | 9/9 | 1.12x | yes | 2 |
| ssve_reassocfix | 80.201 | 78.453 | 9/9 | 1.16x | yes | 2 |

#### softmax torchx64x8192 on sve (device)

| variant | median ms | min ms | reps ok/total | speedup vs main | all match | smstart in asm |
|---|---:|---:|---:|---:|---|---:|
| main | 107.374 | 102.108 | 9/9 | 1.00x | yes | 0 |
| pr14 | 99.843 | 97.085 | 9/9 | 1.08x | yes | 0 |
| pr14fix | 103.569 | 98.375 | 9/9 | 1.04x | yes | 0 |
| reassocfix | 100.385 | 84.605 | 9/9 | 1.07x | yes | 0 |

#### softmax tritonx256x2048 on sve (device)

| variant | median ms | min ms | reps ok/total | speedup vs main | all match | smstart in asm |
|---|---:|---:|---:|---:|---|---:|
| main | 92.454 | 87.385 | 9/9 | 1.00x | yes | 0 |
| pr14 | 83.863 | 79.983 | 9/9 | 1.10x | yes | 0 |
| pr14fix | 82.266 | 78.308 | 9/9 | 1.12x | yes | 0 |
| reassocfix | 81.262 | 78.846 | 9/9 | 1.14x | yes | 0 |

#### softmax tritonx4x8192 on sve (device)

| variant | median ms | min ms | reps ok/total | speedup vs main | all match | smstart in asm |
|---|---:|---:|---:|---:|---|---:|
| main | 8.597 | 8.068 | 9/9 | 1.00x | yes | 0 |
| pr14 | 7.582 | 7.195 | 9/9 | 1.13x | yes | 0 |
| pr14fix | 7.378 | 7.357 | 9/9 | 1.17x | yes | 0 |
| reassocfix | 7.346 | 7.230 | 9/9 | 1.17x | yes | 0 |

#### softmax tritonx64x8192 on sve (device)

| variant | median ms | min ms | reps ok/total | speedup vs main | all match | smstart in asm |
|---|---:|---:|---:|---:|---|---:|
| main | 89.795 | 86.539 | 9/9 | 1.00x | yes | 0 |
| pr14 | 83.060 | 79.708 | 9/9 | 1.08x | yes | 0 |
| pr14fix | 81.380 | 79.313 | 9/9 | 1.10x | yes | 0 |
| reassocfix | 81.624 | 78.008 | 9/9 | 1.10x | yes | 0 |

## Appendix: files

- REPORT.md (this file)
- A_qemu/, B_qemu/, main_qemu/: qemu suites
- A_qprof/: qemu instruction-count runs
- A_phone/: phone suites
- A_bench/, B_bench/, B_bench_reps9/, main_bench/: phone timing sessions (raw/ has every run)
- B_mca/: llvm-mca runs
- matrix/: Canoe three-build matrix (sweep.txt, verify.txt, ir_identity.txt, summary.md, per-label artifacts)
- hawi/: HAWI run (matrix/, pytest_A/, pytest_main/, probe.txt, cpuinfo.txt)
- resume_progress.log, progress.log, phone_progress.log: orchestrator logs
- steps/: per-step outputs of the resumable runner
