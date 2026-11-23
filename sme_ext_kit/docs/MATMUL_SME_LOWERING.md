# How a Triton matmul reaches SME: blocks → K strips → packed tiles → ZA chunks

A reference for how cpu-mlir compiles a matmul for `--target=sme`. It goes from the Python
kernel down to `fmopa`, with pseudocode, the real IR at each stage, and the commands that show it.
It also covers what the `sme-outline-ext` work changed (see also `docs/SME_OUTLINE_EXT.md`).

Running example: `triton_shared_examples/test_matmul.py`, **C = A × B with A 179×167, B 167×321**
(f32), SME, SVL 128 bits unless stated otherwise.

---

## 0. The picture in one place

There are four levels of splitting. Only the last depends on the hardware:

| level | who does it | size | runs as |
|---|---|---|---|
| 1. blocks of C | Triton (the grid) | 32×64 per program, fixed by `BLOCK_SIZE_M/N` | 36 independent programs |
| 2. K strips | the kernel's k-loop | 16 along K (`BLOCK_SIZE_K`) | 11 iterations per program |
| 3. packed tiles | `TritonARM64MatmulToMmt4D` (pack + `linalg.mmt4d`) | 32×32×1, fixed (`blockFactors {32,32,1}`) | a change of memory layout |
| 4. ZA chunks | `TritonARM64LinalgTiling` + vectorization + ArmSME | 8·vscale × 8·vscale = **all of ZA** | 4 `fmopa` (za0–za3) per k |

```
C 179x321 ──(1)──► 32x64 block ──(2)──► += A(32x16) x B(16x64), 11 times
                                              │
                                     (3) pack: 32x64 → 2 tiles of 32x32
                                              │
                                     (4) each 32x32 tile → chunks of 8vs x 8vs
                                              │       (= za0..za3, 4vs x 4vs each)
                                              ▼
                                  per k: 2 loads + 4 fmopa
```
vscale = SVL / 128. At SVL 128, vscale = 1.

---

## 1. Level 1: blocks (the Triton grid)

The kernel is written for one 32×64 block of C. The grid starts one program per block:

```
programs = cdiv(179, 32) x cdiv(321, 64) = 6 x 6 = 36
```
```
                 C (179 x 321)
     ┌──────┬──────┬──────┬──────┬──────┬──────┐
     │ P0   │ P1   │ P2   │ P3   │ P4   │ P5   │   one box = one program = one 32x64 block
     ├──────┼──────┼──────┼──────┼──────┼──────┤
     │ P6   │ ...  │      │      │      │      │   the last row/column of boxes sticks out
     ├──────┼──────┼──────┼──────┼──────┼──────┤   past the matrix: those elements are masked
     │ ...  │      │      │      │      │      │
     └──────┴──────┴──────┴──────┴──────┴──────┘
```
The launcher (`launcher_*.cpp`) runs the 36 programs one after another; each is one call of
`matmul_kernel(..., pid)`. Everything below happens **inside one program**.

## 2. Level 2: K strips (the k-loop) and the load buffers

### The source
```python
accumulator = tl.zeros((32, 64), dtype=tl.float32)
for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):                          # cdiv(167, 16) = 11
    a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)   # 32x16 strip of A
    b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)   # 16x64 strip of B
    accumulator += tl.dot(a, b)
    a_ptrs += BLOCK_SIZE_K * stride_ak      # move 16 columns right in A
    b_ptrs += BLOCK_SIZE_K * stride_bk      # move 16 rows down in B
tl.store(c_ptrs, accumulator, mask=c_mask)
```
```
 A strip (32x16)      B strip (16x64)            accumulator (32x64)
 ┌──┐                 ┌────────────────┐         ┌────────────────┐
 │k0│            x    │ k0             │    +=   │ the SAME block │
 │  │                 └────────────────┘         │ every iteration│
 └──┘                                            └────────────────┘
 next iteration: next 16 columns of A, next 16 rows of B
```
The three facts everything else depends on:
1. 36 programs × 11 iterations = **396 k-iterations**.
2. **A and B strips are new data** every iteration.
3. **C is the same block** every iteration, only added to.

### What the compiler makes of it (`-ttshared.mlir`, triton-shared's output)
```mlir
%1 = linalg.fill 0.0 -> tensor<32x64xf32>                                   // accumulator = 0
%r:3 = scf.for %k = 0 to 11 iter_args(%offA = .., %offB = .., %acc = %1)
         -> (index, index, tensor<32x64xf32>) {                            // the loop's RESULTS
  %alloc   = memref.alloc() : memref<32x16xf32>      // scratch buffer for the A strip
  linalg.fill 0.0 -> %alloc                          //   (only if masked) zeros for the edge
  memref.copy A[valid part] -> %alloc[valid part]    //   the in-bounds data
  %alloc_9 = memref.alloc() : memref<16x64xf32>      // same for the B strip
  ...
  %d = linalg.matmul ins(%a, %b) outs(%acc)          // tl.dot + the +=
  scf.yield %offA + 16, %offB + 16*N, %d
}
```
- **The load buffers** (`%alloc`, `%alloc_9`) are scratch copies of the current A and B strips,
  zero-padded where the mask is false. A strip is 2 KB, a B strip 4 KB. The buffer deallocation
  pipeline adds the matching `free`s inside the loop.
- **The loop has results:** the two pointer offsets (`index`) and the accumulator. After
  bufferization the accumulator lives in memory, but the two `index` results remain. That matters
  for outlining (section 6).

Pseudocode of one program at this point:
```
acc = zeros(32x64)
offA, offB = start
for k in 0..10:
    bufA = malloc(32x16); bufA[:] = 0; copy valid A strip into bufA
    bufB = malloc(16x64); bufB[:] = 0; copy valid B strip into bufB
    acc += bufA @ bufB
    free(bufA); free(bufB)
    offA += 16; offB += 16*N
masked store acc -> C
```

## 3. Level 3: packing and `linalg.mmt4d` (`TritonARM64MatmulToMmt4D`)

### What packing is
Packing cuts a matrix into fixed-size tiles and stores each tile contiguously: a copy plus a
transpose. A small example, a 4×4 matrix with 2×2 tiles (the real tile is 32):
```
 row-major 4x4                packed: X4[tileRow][tileCol][r][c]   (shape 2x2x2x2)
  x00 x01 │ x02 x03           tile(0,0) = x00 x01 x10 x11
  x10 x11 │ x12 x13           tile(0,1) = x02 x03 x12 x13
  ────────┼────────           tile(1,0) = x20 x21 x30 x31
  x20 x21 │ x22 x23           tile(1,1) = x22 x23 x32 x33
  x30 x31 │ x32 x33
```
4×4 with 2×2 tiles = **4 tiles** (2 × 2 of them), not 2. Unpacking is the inverse.

### The real tile sizes
`ARM64MatmulToMmt4DPass.cpp`: `options.blockFactors = {32, 32, 1}` → M0 = 32, N0 = 32, K0 = 1
(fixed, with a "TODO: figure out optimal block sizes"). Upstream `populateBlockPackMatmulPatterns`
turns `linalg.matmul` into `pack(A), pack(B), pack(C) → mmt4d → unpack(C)`:

| operand | normal | packed | layout |
|---|---|---|---|
| A strip | 32×16 | `1x16x32x1` | `[M/32][K/1][32][1]`: for each k, a run of 32 values (one column of A) |
| B strip | 16×64 | `2x16x32x1` | `[N/32][K/1][32][1]`: B **transposed**: for each k, a run of 32 values (one row of B) |
| C block | 32×64 | `1x2x32x32` | `[M/32][N/32][32][32]`: two 32×32 tiles |

```
mmt4d:   C4[m][n][i][j] += Σ_k Σ_kk  A4[m][k][i][kk] * B4[n][k][j][kk]       (kk = 0 only, K0 = 1)
```
With K0 = 1, each k is **one column of A (32) times one row of B (32), added into a 32×32 tile
of C**: an **outer product**. Sixteen of them make (32×16) × (16×32).
```
 column of A (32)     row of B (32)              32x32 update of one C tile
      a0                                       a0*b0   a0*b1  ...  a0*b31
      a1         x    b0 b1 ... b31     =      a1*b0   ...
      ...                                      ...
      a31                                      a31*b0  ...         a31*b31
```

**Odd sizes.** A pack pads with zeros up to a multiple of the tile size. A plain 179×167×321
matmul (the torch path, no Triton blocks) becomes M 192 (6 tiles), N 352 (11 tiles), K 167
(K0 = 1: no padding). In Triton the blocks are always a static 32×64×16 and the masked loads
already put zeros at the edges, so its packs never pad.

### How the packs are lowered
The pass then lowers each pack/unpack into reshapes and transposes (`linalg::lowerPack` /
`lowerUnPack`). Real IR after the pass, with the hoist (section 7.2):
```mlir
// pack C ONCE, before the loop: 32x64 -> 1x32x2x32 -> transpose -> 1x2x32x32
%expanded   = tensor.expand_shape %1 [[0, 1], [2, 3]] output_shape [1, 32, 2, 32]
%transposed = linalg.transpose ins(%expanded) outs(tensor<1x2x32x32xf32>) permutation = [0, 2, 1, 3]
%30:3 = scf.for ... iter_args(%offA, %offB, %arg18 = %transposed) -> (index, index, tensor<1x2x32x32xf32>) {
  // pack A strip: 32x16 -> 1x32x16x1 -> 1x16x32x1
  %transposed_18 = linalg.transpose ins(expand(%a) : tensor<1x32x16x1xf32>) outs(tensor<1x16x32x1xf32>) permutation = [0, 2, 1, 3]
  // pack B strip: 16x64 -> 16x1x2x32 -> 2x16x32x1
  %transposed_20 = linalg.transpose ins(expand(%b) : tensor<16x1x2x32xf32>) outs(tensor<2x16x32x1xf32>) permutation = [2, 0, 3, 1]
  %70 = linalg.mmt4d ins(%transposed_18, %transposed_20) outs(%arg18 : tensor<1x2x32x32xf32>)
  scf.yield ..., %70                                   // C stays packed
}
// unpack C ONCE, after the loop: 1x2x32x32 -> 1x32x2x32 -> 32x64
%transposed_0 = linalg.transpose ins(%30#2) outs(tensor<1x32x2x32xf32>) permutation = [0, 2, 1, 3]
%collapsed    = tensor.collapse_shape %transposed_0 [[0, 1], [2, 3]] : tensor<1x32x2x32xf32> into tensor<32x64xf32>
```
On main (without the hoist), the C transpose and the unpack transpose + collapse + copy were
**inside** the loop, so every k-iteration converted C to the packed layout and back.

## 4. Level 4: ZA chunks (tiling, vectorization, ArmSME)

### The hardware
- **ZA** is SME's 2-D register array. For f32 it is split into **4 tiles, `za0.s`–`za3.s`**,
  each **(SVL/32) × (SVL/32)** = 4·vscale × 4·vscale.
- **`fmopa za<i>.s, z_a, z_b`** adds the outer product of two vectors (each SVL/32 floats) into
  one ZA tile: exactly the mmt4d step from section 3.
- ZA instructions only work in **streaming mode** (section 6).

### Tiling (`TritonARM64LinalgTiling`, `tileMmt4DOp`)
The mmt4d `[1x2 tiles][32x32 each]` is tiled with sizes `[1, 1, 0, 8·vscale, 8·vscale, 0]` (the M
and N tile dims), then a second step tiles K by 1:
```mlir
scf.for %m = 0 to 1                          // tile row of C   (M/32)
 scf.for %n = 0 to 2                         // tile column     (N/32)
  scf.for %i = 0 to 32 step 8*vscale         // chunk row inside the 32x32 tile
   scf.for %j = 0 to 32 step 8*vscale        // chunk column
     %a = extract_slice A4[%m, 0, %i, 0] [1, 16, 8vs, 1]
     %b = extract_slice B4[%n, 0, %j, 0] [1, 16, 8vs, 1]
     %c = extract_slice C4[%m, %n, %i, %j] [1, 1, 8vs, 8vs]
     scf.for %k = 0 to 16 step 1             // the K strip, one outer product each
       linalg.mmt4d ins(%a[k] : 1x1x?x1, %b[k] : 1x1x?x1) outs(%c : 1x1x?x?)
```

### Vectorization (`TritonARM64LinalgVectorization`) and lowering (`TritonARM64VectorLowering`)
The mmt4d (one k step) with vector sizes `{1,1,1,[8],[8],1}` becomes: read the A piece and the B
piece, each broadcast to `[8]x[8]` (`permutation_map`), multiply elementwise, and sum into the C
chunk. No masks, because `assumeDynamicDimsMatchVecSizes = true`:
```mlir
%124 = vector.transfer_read %Apiece ... permutation_map = #map20 : tensor<1x1x?x1xf32>, vector<1x1x1x[8]x[8]x1xf32>
%126 = vector.transfer_read %Bpiece ... permutation_map = #map21 : tensor<1x1x?x1xf32>, vector<1x1x1x[8]x[8]x1xf32>
%128 = vector.transfer_read %Cchunk ...                          : tensor<1x1x?x?xf32>, vector<1x1x[8]x[8]xf32>
%129 = arith.mulf %124, %126 : vector<1x1x1x[8]x[8]x1xf32>
%130 = vector.multi_reduction <add>, %129, %128 [2, 5] : vector<1x1x1x[8]x[8]x1xf32> to vector<1x1x[8]x[8]xf32>
```
The vector lowering turns this into `vector.outerproduct`, with the C chunk carried in a
register across the 16 k steps. Real IR after `TritonARM64VectorLowering`:
```mlir
%84 = vector.transfer_read %Cchunk[...] : memref<?x?xf32>, vector<[8]x[8]xf32>   // C chunk in
%85 = scf.for %k = 0 to 16 iter_args(%acc = %84) -> (vector<[8]x[8]xf32>) {
  %86 = vector.transfer_read %Acol[...] : vector<[8]xf32>                       // column piece of A
  %87 = vector.transfer_read %Brow[...] : vector<[8]xf32>                       // row piece of B
  %88 = vector.outerproduct %86, %87, %acc {kind = #vector.kind<add>}
  scf.yield %88
}
vector.transfer_write %85, %Cchunk[...] : vector<[8]x[8]xf32>                   // C chunk out
```

### To ZA (`VectorLegalization`, `ConvertVectorToArmSME`, `ConvertArmSMEToSCF`)
One `vector<[8]x[8]xf32>` is 2×2 ZA tiles of `[4]x[4]`, so legalization splits it into 4:
```
 one 8vs x 8vs chunk = ALL of ZA        per k:  a = A column piece (8vs),  b = B row piece (8vs)
 ┌─────────┬─────────┐
 │  za0.s  │  za1.s  │   fmopa za0.s, a[0:4vs]   x b[0:4vs]
 │ 4vs×4vs │ 4vs×4vs │   fmopa za1.s, a[0:4vs]   x b[4vs:8vs]
 ├─────────┼─────────┤   fmopa za2.s, a[4vs:8vs] x b[0:4vs]
 │  za2.s  │  za3.s  │   fmopa za3.s, a[4vs:8vs] x b[4vs:8vs]
 └─────────┴─────────┘
```
Op counts: after `ConvertVectorToArmSME` there are 4 `arm_sme.tile_load`, 4 `arm_sme.outerproduct`
and 4 store-tile-slice loops. The assembly of `matmul_kernel_sme_0`:
```asm
.LBB1_39:  ld1w {za0h.s[w14, 0]}, p1/z, [x15]     // load the C chunk into za0, one row at a time
           ...                                    // same for za1, za2, za3
.LBB1_46:  ld1w  { z0.s }, p1/z, [...]            // A column piece, first half
           ld1w  { z1.s }, p1/z, [...]            // B row piece, first half
           fmopa za0.s, p1/m, p1/m, z0.s, z1.s
           fmopa za1.s, p1/m, p1/m, z0.s, z3.s
           fmopa za2.s, p1/m, p1/m, z2.s, z1.s
           fmopa za3.s, p1/m, p1/m, z2.s, z3.s
           ...                                    // 16 k steps: 16 fmopa per ZA tile in total
           st1w  {za0h.s[...]}, ...               // store the chunk back
```

### How many chunks per packed tile, by SVL
The packed tile is always 32×32; the chunk is 8·vscale × 8·vscale:

| SVL | vscale | one ZA f32 tile | chunk (all 4 tiles) | chunks per 32×32 tile |
|---|---|---|---|---|
| 128 | 1 | 4×4 | 8×8 | 4 × 4 = 16 |
| 256 | 2 | 8×8 | 16×16 | 2 × 2 = 4 |
| 512 | 4 | 16×16 | 32×32 | 1 |
| 1024 | 8 | 32×32 | 64×64 > 32×32 | **breaks** (below) |

```
SVL 128: one 32x32 packed tile          SVL 512: one 32x32 packed tile
┌──┬──┬──┬──┐                           ┌─────────────┐
│8²│8²│8²│8²│  16 chunks; each:         │ za0  │ za1  │   1 chunk = the whole tile
├──┼──┼──┼──┤  load C chunk into ZA,    │──────┼──────│
│  │  │  │  │  16 k steps x 4 fmopa,    │ za2  │ za3  │
├──┼──┼──┼──┤  store ZA back            └─────────────┘
│  │  │  │  │
└──┴──┴──┴──┘
```

**Known bug, pre-existing (also with main's behaviour): SVL ≥ 1024 bits aborts** (`free(): invalid
next size`, a heap overflow). The likely cause, not yet proven: the 64×64 chunk is bigger than the 32×32
packed tile, and the mmt4d vectorization sets `assumeDynamicDimsMatchVecSizes = true`
(`ARM64VectorizationPass.cpp`), so the accesses are not masked and run past the buffer. Current
hardware has SVL ≤ 512. A fix would cap the chunk at the packed tile size or make the pack size follow SVL.
```bash
cd runs/sme_ext/svl_test; source ../../../env-branch.sh
for svl in 16 32 64 128 256; do TRITON_ALWAYS_COMPILE=1 QEMU_CPU=max,sme_fa64=off,sme-default-vector-length=$svl \
  python ../../../tools/sme_ext/run_matmul.py sme 179 167 321 2>&1 | tail -1; done     # bytes: 16=128 bits ... 256=2048 bits
```

## 5. The whole nest for one program, as pseudocode (this branch, SVL 128)

```
# matmul_kernel(pid)                                          [normal mode]
bufA = malloc(32x16); bufB = malloc(16x64)                    # load buffers, once (7.3)
packA = alloc(1x16x32x1); packB = alloc(2x16x32x1)
C4 = pack(zeros(32x64))                                       # 1x2x32x32, once (7.2)
offA, offB = call matmul_kernel_sme_0(...)                    # (7.1)
    # ---- matmul_kernel_sme_0                                [streaming + ZA]
    for k in 0..10:                                           # level 2: K strips
        bufA = 0; copy valid A strip -> bufA                  # masked load
        bufB = 0; copy valid B strip -> bufB
        packA = transpose(bufA)                               # level 3: new data, packed every k
        packB = transpose(bufB)
        for n in 0..1:                                        # level 3: the 2 packed C tiles
          for i in 0..31 step 8:                              # level 4: chunks (8 = 8*vscale)
            for j in 0..31 step 8:
                ZA[za0..za3] = C4[0][n][i:i+8][j:j+8]         # ld1w za*h.s rows
                for kk in 0..15:                              # one outer product per kk
                    a = packA[0][kk][i:i+8]; b = packB[n][kk][j:j+8]
                    fmopa za0 += a[0:4] x b[0:4];  fmopa za1 += a[0:4] x b[4:8]
                    fmopa za2 += a[4:8] x b[0:4];  fmopa za3 += a[4:8] x b[4:8]
                C4[0][n][i:i+8][j:j+8] = ZA                   # st1w za*h.s rows
        offA += 16; offB += 16*N
    return offA, offB
    # ---- back in normal mode
acc = unpack(C4)                                              # once (7.2)
masked store acc -> C                                         # memrefCopy (still slow)
free everything
```
What remains to improve, in this order:
- **C goes in and out of ZA** once per k-iteration (the `ZA = C4[...]` / `C4[...] = ZA` lines).
  With SVL 512 and one chunk per tile, C could stay in ZA for the whole k-loop.
- **The final store of C** goes through the runtime `memrefCopy`: 0.9M of 31.9M instructions.

## 6. Streaming mode and the outline pass

- `fmopa` and ZA work only after **`smstart`** (streaming mode + ZA on). In streaming mode, many
  normal instructions (NEON, gathers) are illegal on CPUs without FEAT_SME_FA64 (SIGILL). A call to a
  normal function (`malloc`, `free`) must **`smstop`**, call, and **`smstart`** again, saving and
  restoring ZA (`__arm_tpidr2_save/restore`) around it. Each switch is expensive on hardware.
- Upstream **`EnableArmStreaming`** marks every function that contains ZA ops as streaming, as a
  whole. That's why **`TritonARM64SMEFuncOutline`** exists: it moves each outermost loop that uses
  ZA into its own private function `<fn>_sme_N`, so only that function is streaming. It runs after
  the inliner, right before the ArmSME conversion (pass order: section 8).

```
before outlining                          after outlining
matmul_kernel  [streaming, all of it]     matmul_kernel  [normal]
   setup                                     setup
   k-loop with ZA                            offA, offB = call matmul_kernel_sme_0(...)  ──► [streaming]: the k-loop
   store C                                   store C
```

## 7. What `sme-outline-ext` changed (main → this branch)

### 7.1 Outline loops that have results (`ARM64SMEFuncOutlinePass.cpp`)
Main refused any loop with results ("after bufferization the loop writes to memrefs and has no
results"). The Triton k-loop returns `(index, index)` (the offsets), so it was refused, and the
**whole kernel** stayed streaming, with the malloc/free of section 2 inside it. The debug line:
```
[triton-arm64-sme-func-outline] could not outline: %37:2 = scf.for ... -> (index, index)
```
Now the outlined function returns the loop's results and the call's results replace them (only
scalable-vector results are still refused). Plus a launcher fix: `doesFuncReturnValue`
(`LowerTritonLinalgToLLVM.cpp`) only looks at public functions, otherwise the private
helper returning values made the Triton kernel look like a torch kernel.

### 7.2 Hoist the accumulator pack/unpack (`ARM64MatmulToMmt4DPass.cpp`, `hoistPackedAccumulator`)
It matches an `scf.for` iter_arg whose only uses are `linalg.pack %acc` and the destination of
the `linalg.unpack` that is yielded. It carries the packed tensor instead: pack the init before
the loop, unpack the result after it. It runs between block packing and pack lowering, so the pass
now applies its patterns in two rounds. It refuses padding, dynamic shapes, and mismatching tiles or perms.
```
main:    for k { pack A; pack B; pack C; mmt4d; unpack C }      x396
branch:  pack C; for k { pack A; pack B; mmt4d }; unpack C
```

### 7.3 Allocate the load buffers once (`OptimizeTritonShared.cpp`)
`bufferization::createBufferLoopHoistingPass()` before the deallocation pipeline: static-size
allocs in a loop move before it.
```
main:    for k { malloc bufA; malloc bufB; ...; free; free }   x396, each call smstop/smstart in streaming code
branch:  malloc bufA; malloc bufB; for k { ... }; free; free
```

### Results (qemu, FA64 off, SVL 128)
| | main | branch |
|---|---|---|
| kernel instructions | 41.9M | 31.9M (mostly 7.2) |
| `smstart` executed | 2772 | 144 (7.1 + 7.3) |
| whole public kernel streaming | yes | no |
```bash
bash tools/sme_ext/ab_matmul.sh runs/sme_ext/mm     # base / hoist / outline / both / all
```

## 8. See every stage yourself

```bash
cd ~/thats_me/triton-mlir-arm && source env-branch.sh && export QEMU_CPU=max,sme_fa64=off
bash tools/sme_ext/ab_matmul.sh runs/sme_ext/mm                      # makes runs/sme_ext/mm/<variant>/matmul_kernel-*.mlir/.S
cd runs/sme_ext/mm/all
arm64-mlir-opt --vector-ext=sme -optimize-triton-shared -mlir-print-ir-after-all -mlir-disable-threading \
  -mlir-elide-elementsattrs-if-larger=16 matmul_kernel-ttshared.mlir -o /dev/null 2> after_all.mlir
grep -n "IR Dump After" after_all.mlir        # pass order
```
Pass order (SME), and what to look for after each:

| pass | look for |
|---|---|
| `TritonARM64MatmulToMmt4D` | `linalg.mmt4d`, the pack/unpack transposes, where they sit relative to `scf.for` (levels 2–3) |
| `TritonARM64LinalgTiling` | the chunk loops `step %c8_vscale` and the k loop `0 to 16 step 1` (level 4) |
| `TritonARM64LinalgVectorization` | broadcast `transfer_read`s + `arith.mulf` + `vector.multi_reduction <add>` into `vector<1x1x[8]x[8]xf32>` |
| `TritonARM64VectorLowering` | `vector.outerproduct` inside `scf.for 0 to 16`, the chunk read/written around it |
| `BufferLoopHoistingPass` | `memref.alloc 32x16 / 16x64` before the loop (7.3) |
| `InlinerPass`, `TritonARM64SMEFuncOutline` | `func.func private @matmul_kernel_sme_0(...) -> (index, index)` (7.1) |
| `VectorLegalization`, `ConvertVectorToArmSMEPass` | 4 `arm_sme.tile_load` / `arm_sme.outerproduct` (za0–za3) |
| `EnableArmStreaming` | `arm_streaming` + `arm_new_za` only on `_sme_0` |

Main's behaviour, from the same build: prefix with
`TRITON_ARM64_NO_PACK_HOIST=1 TRITON_ARM64_NO_OUTLINE_RESULTS=1 TRITON_ARM64_NO_BUFFER_HOIST=1`
(temporary switches; they go away before the PR).
```bash
awk '/^matmul_kernel_sme_0:/,/^.Lfunc_end/' matmul_kernel-arm64.S | grep -E "fmopa|za[0-3]h|smstart|smstop|\bbl\b" | head -40
```
