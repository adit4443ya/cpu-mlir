// RUN: arm64-mlir-opt %s -split-input-file -triton-arm64-sme-func-outline | FileCheck %s
// RUN: arm64-mlir-opt %s -split-input-file -triton-arm64-sme-func-outline="ssve-reductions=1" | FileCheck %s --check-prefix=SSVE

// The outermost loop of a ZA tile loop nest moves into a private, internal
// function: one call per transpose. Only the buffers become arguments; vscale
// and the index arithmetic on it are recomputed inside, where they follow the
// streaming vector length.
// CHECK-LABEL: func.func @outline_tile_loops(
// CHECK-SAME:    %[[SRC:[^:]+]]: memref<64x64xf32>, %[[DST:[^:]+]]: memref<64x64xf32>)
// CHECK-NOT:     scf.for
// CHECK:         call @outline_tile_loops_sme_0(%[[SRC]], %[[DST]]) : (memref<64x64xf32>, memref<64x64xf32>) -> ()
// CHECK-NEXT:    return
// CHECK-LABEL: func.func private @outline_tile_loops_sme_0(
// CHECK-SAME:    memref<64x64xf32>, %{{[^:]+}}: memref<64x64xf32>)
// CHECK-SAME:    llvm.linkage = #llvm.linkage<internal>
// CHECK:         %[[VS:.+]] = vector.vscale
// CHECK:         %[[STEP:.+]] = arith.muli %[[VS]], %{{.+}} : index
// CHECK:         scf.for {{.+}} step %[[STEP]]
// CHECK:           scf.for {{.+}} step %[[STEP]]
// CHECK:             vector.transfer_read {{.+}} : memref<64x64xf32>, vector<[4]x[4]xf32>
// CHECK:             vector.transfer_write {{.+}} : vector<[4]x[4]xf32>, memref<64x64xf32>
// CHECK:         return
// CHECK-NOT:   func.func private @outline_tile_loops_sme_1
func.func @outline_tile_loops(%src: memref<64x64xf32>, %dst: memref<64x64xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c64 = arith.constant 64 : index
  %pad = arith.constant 0.0 : f32
  %vscale = vector.vscale
  %step = arith.muli %vscale, %c4 : index
  scf.for %i = %c0 to %c64 step %step {
    scf.for %j = %c0 to %c64 step %step {
      %t = vector.transfer_read %src[%j, %i], %pad {permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<64x64xf32>, vector<[4]x[4]xf32>
      vector.transfer_write %t, %dst[%i, %j] : vector<[4]x[4]xf32>, memref<64x64xf32>
    }
  }
  return
}

// -----

// Two ZA loop nests in one function get one function each; the SVE loop
// between them (1-D scalable vectors only) stays in the caller.
// CHECK-LABEL: func.func @two_tile_loops(
// CHECK:         call @two_tile_loops_sme_0(
// CHECK:         scf.for
// CHECK:           vector.transfer_read {{.+}} vector<[4]xf32>
// CHECK:         call @two_tile_loops_sme_1(
// CHECK:         return
// CHECK-DAG:   func.func private @two_tile_loops_sme_0(
// CHECK-DAG:   func.func private @two_tile_loops_sme_1(
func.func @two_tile_loops(%a: memref<64x64xf32>, %b: memref<64x64xf32>, %c: memref<64xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c64 = arith.constant 64 : index
  %pad = arith.constant 0.0 : f32
  %vscale = vector.vscale
  %step = arith.muli %vscale, %c4 : index
  scf.for %i = %c0 to %c64 step %step {
    %t = vector.transfer_read %a[%c0, %i], %pad {permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<64x64xf32>, vector<[4]x[4]xf32>
    vector.transfer_write %t, %b[%i, %c0] : vector<[4]x[4]xf32>, memref<64x64xf32>
  }
  scf.for %i = %c0 to %c64 step %step {
    %v = vector.transfer_read %c[%i], %pad : memref<64xf32>, vector<[4]xf32>
    %w = arith.addf %v, %v : vector<[4]xf32>
    vector.transfer_write %w, %c[%i] : vector<[4]xf32>, memref<64xf32>
  }
  scf.for %i = %c0 to %c64 step %step {
    %t = vector.transfer_read %b[%c0, %i], %pad {permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<64x64xf32>, vector<[4]x[4]xf32>
    vector.transfer_write %t, %a[%i, %c0] : vector<[4]x[4]xf32>, memref<64x64xf32>
  }
  return
}

// -----

// No ZA tiles (only 1-D scalable SVE vectors): nothing is outlined.
// CHECK-LABEL: func.func @no_tiles(
// CHECK-NOT:     call
// CHECK:         scf.for
// CHECK-NOT:   func.func private
func.func @no_tiles(%c: memref<64xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c64 = arith.constant 64 : index
  %pad = arith.constant 0.0 : f32
  %vscale = vector.vscale
  %step = arith.muli %vscale, %c4 : index
  scf.for %i = %c0 to %c64 step %step {
    %v = vector.transfer_read %c[%i], %pad : memref<64xf32>, vector<[4]xf32>
    vector.transfer_write %v, %c[%i] : vector<[4]xf32>, memref<64xf32>
  }
  return
}

// -----

// A vector loaded outside the loop cannot be passed in: entering streaming
// mode clears the vector registers. The loop stays (the whole function will
// become streaming, as without this pass).
// CHECK-LABEL: func.func @vector_from_outside(
// CHECK-NOT:     call
// CHECK:         vector.load
// CHECK:         scf.for
// CHECK-NOT:   func.func private
func.func @vector_from_outside(%a: memref<64x64xf32>, %b: memref<64x64xf32>, %c: memref<64xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c64 = arith.constant 64 : index
  %pad = arith.constant 0.0 : f32
  %vscale = vector.vscale
  %step = arith.muli %vscale, %c4 : index
  %row = vector.load %c[%c0] : memref<64xf32>, vector<[4]xf32>
  scf.for %i = %c0 to %c64 step %step {
    %t = vector.transfer_read %a[%c0, %i], %pad {permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<64x64xf32>, vector<[4]x[4]xf32>
    vector.transfer_write %t, %b[%i, %c0] : vector<[4]x[4]xf32>, memref<64x64xf32>
    vector.store %row, %c[%i] : memref<64xf32>, vector<[4]xf32>
  }
  return
}

// -----

// A buffer sized from vscale outside would be sized for the non-streaming
// vector length, but used with the streaming one inside: not outlined.
// CHECK-LABEL: func.func @buffer_sized_by_vscale(
// CHECK-NOT:     call
// CHECK:         memref.alloc
// CHECK:         scf.for
// CHECK-NOT:   func.func private
func.func @buffer_sized_by_vscale(%a: memref<64x64xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c64 = arith.constant 64 : index
  %pad = arith.constant 0.0 : f32
  %vscale = vector.vscale
  %step = arith.muli %vscale, %c4 : index
  %buf = memref.alloc(%step, %step) : memref<?x?xf32>
  scf.for %i = %c0 to %c64 step %step {
    %t = vector.transfer_read %a[%c0, %i], %pad {permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<64x64xf32>, vector<[4]x[4]xf32>
    vector.transfer_write %t, %buf[%c0, %c0] : vector<[4]x[4]xf32>, memref<?x?xf32>
  }
  memref.dealloc %buf : memref<?x?xf32>
  return
}

// -----

// A loop with results, e.g. the pointer offsets a Triton k-loop carries: the
// outlined function returns them and the call's results replace the loop's.
// CHECK-LABEL: func.func @tile_loop_with_results(
// CHECK:         %[[R:.+]]:2 = call @tile_loop_with_results_sme_0(%{{.+}}, %{{.+}}, %{{.+}}) : (memref<64x64xf32>, memref<64x64xf32>, index) -> (index, i32)
// CHECK:         return %[[R]]#0, %[[R]]#1 : index, i32
// CHECK-LABEL: func.func private @tile_loop_with_results_sme_0(
// CHECK-SAME:    -> (index, i32)
// CHECK:         %[[L:.+]]:2 = scf.for
// CHECK:         return %[[L]]#0, %[[L]]#1 : index, i32
func.func @tile_loop_with_results(%src: memref<64x64xf32>, %dst: memref<64x64xf32>, %off: index) -> (index, i32) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c64 = arith.constant 64 : index
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %pad = arith.constant 0.0 : f32
  %vscale = vector.vscale
  %step = arith.muli %vscale, %c4 : index
  %r:2 = scf.for %i = %c0 to %c64 step %step iter_args(%o = %off, %n = %c0_i32) -> (index, i32) {
    %t = vector.transfer_read %src[%c0, %i], %pad {permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<64x64xf32>, vector<[4]x[4]xf32>
    vector.transfer_write %t, %dst[%i, %c0] : vector<[4]x[4]xf32>, memref<64x64xf32>
    %o2 = arith.addi %o, %c64 : index
    %n2 = arith.addi %n, %c1_i32 : i32
    scf.yield %o2, %n2 : index, i32
  }
  return %r#0, %r#1 : index, i32
}

// -----

// ssve-reductions: a reduction loop over scalable vectors moves into a locally
// streaming function (it switches mode itself, so the caller sets up the
// arguments in normal mode). Its fixed-size accumulator is returned. Without
// the option nothing happens: there are no ZA tiles.
// CHECK-LABEL: func.func @ssve_reassoc_sum(
// CHECK:         scf.for
// CHECK-NOT:   func.func private
// SSVE-LABEL:  func.func @ssve_reassoc_sum(
// SSVE:          %[[R:.+]] = call @ssve_reassoc_sum_ssve_0(%{{.+}}) : (memref<1024xf32>) -> vector<1xf32>
// SSVE:          return %[[R]]
// SSVE-LABEL:  func.func private @ssve_reassoc_sum_ssve_0(
// SSVE-SAME:     -> vector<1xf32>
// SSVE-SAME:     llvm.arm_locally_streaming
// SSVE:          vector.vscale
// SSVE:          scf.for
// SSVE:            vector.reduction <add>, %{{.+}}, %{{.+}} fastmath<reassoc> : vector<[4]xf32> into f32
func.func @ssve_reassoc_sum(%in: memref<1024xf32>) -> vector<1xf32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1024 = arith.constant 1024 : index
  %pad = arith.constant 0.0 : f32
  %init = arith.constant dense<0.0> : vector<1xf32>
  %vscale = vector.vscale
  %step = arith.muli %vscale, %c4 : index
  %r = scf.for %i = %c0 to %c1024 step %step iter_args(%acc = %init) -> (vector<1xf32>) {
    %v = vector.transfer_read %in[%i], %pad : memref<1024xf32>, vector<[4]xf32>
    %a = vector.extract %acc[0] : f32 from vector<1xf32>
    %s = vector.reduction <add>, %v, %a fastmath<reassoc> : vector<[4]xf32> into f32
    %b = vector.broadcast %s : f32 to vector<1xf32>
    scf.yield %b : vector<1xf32>
  }
  return %r : vector<1xf32>
}

// -----

// An in-order floating-point sum becomes FADDA, which is illegal in streaming
// mode (llc: "Expanding reductions for scalable vectors is undefined"): such a
// loop stays in the caller. A max (FMAXNMV) is fine.
// SSVE-LABEL:  func.func @ssve_ordered_sum_stays(
// SSVE:          scf.for
// SSVE:            vector.reduction <add>
// SSVE:          call @ssve_ordered_sum_stays_ssve_0(
// SSVE-LABEL:  func.func private @ssve_ordered_sum_stays_ssve_0(
// SSVE:            vector.reduction <maxnumf>
// SSVE-NOT:    func.func private @ssve_ordered_sum_stays_ssve_1
func.func @ssve_ordered_sum_stays(%in: memref<1024xf32>, %out: memref<2xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c1024 = arith.constant 1024 : index
  %pad = arith.constant 0.0 : f32
  %vscale = vector.vscale
  %step = arith.muli %vscale, %c4 : index
  %sum = scf.for %i = %c0 to %c1024 step %step iter_args(%acc = %pad) -> (f32) {
    %v = vector.transfer_read %in[%i], %pad : memref<1024xf32>, vector<[4]xf32>
    %s = vector.reduction <add>, %v, %acc : vector<[4]xf32> into f32
    scf.yield %s : f32
  }
  %max = scf.for %i = %c0 to %c1024 step %step iter_args(%acc = %pad) -> (f32) {
    %v = vector.transfer_read %in[%i], %pad : memref<1024xf32>, vector<[4]xf32>
    %s = vector.reduction <maxnumf>, %v, %acc : vector<[4]xf32> into f32
    scf.yield %s : f32
  }
  memref.store %sum, %out[%c0] : memref<2xf32>
  memref.store %max, %out[%c1] : memref<2xf32>
  return
}
