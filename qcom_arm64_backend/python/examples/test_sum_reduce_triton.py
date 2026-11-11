# ===- test_sum_reduce_triton.py --------------------------------------------===
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  SPDX-License-Identifier: BSD-3-Clause.
#  For more license information:
#    https://github.com/qualcomm/cpu-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import math

import pytest
import torch

import triton
import triton.language as tl

SUM_REDUCTION_CASES = (
    ((128,), (0,), "r1_p2"),
    ((127,), (0,), "r1_ragged"),
    ((8, 128), (0,), "r2_outer"),
    ((8, 128), (1,), "r2_inner"),
    ((7, 127), (1,), "r2_ragged_inner"),
    ((4, 8, 128), (1,), "r3_middle"),
    ((4, 8, 128), (2,), "r3_inner"),
    ((3, 7, 127), (1, 2), "r3_tail_multi_ragged"),
    ((2, 4, 8, 128), (0, 2), "r4_noncontiguous_multi"),
    ((2, 3, 7, 127), (0, 1, 2, 3), "r4_full_ragged"),
)

TRITON_CASES = [
    pytest.param(shape, axes, case_id, id=case_id)
    for shape, axes, case_id in SUM_REDUCTION_CASES
]


@triton.jit
def sum_reduce_rank1_kernel(input_ptr, output_ptr, d0: tl.constexpr,
                            block_size: tl.constexpr):
    offsets = tl.arange(0, block_size)
    value = tl.sum(tl.load(input_ptr + offsets, mask=offsets < d0, other=0.0), axis=0)
    tl.store(output_ptr, value)


@triton.jit
def sum_reduce_blocks_kernel(input_ptr, partial_ptr, n_elements: tl.constexpr,
                             block_size: tl.constexpr):
    block_id = tl.program_id(0)
    offsets = block_id * block_size + tl.arange(0, block_size)
    value = tl.load(input_ptr + offsets, mask=offsets < n_elements, other=0.0)
    tl.store(partial_ptr + block_id, tl.sum(value, axis=0))


@triton.jit
def sum_reduce_rank2_outer_kernel(input_ptr, output_ptr, d0: tl.constexpr,
                                  d1: tl.constexpr, block_size: tl.constexpr):
    cols = tl.program_id(0) * block_size + tl.arange(0, block_size)
    result = tl.zeros((block_size,), tl.float32)
    for row in range(0, d0):
        result += tl.load(input_ptr + row * d1 + cols, mask=cols < d1, other=0.0)
    tl.store(output_ptr + cols, result, mask=cols < d1)


@triton.jit
def sum_reduce_rank2_inner_kernel(input_ptr, output_ptr, d1: tl.constexpr,
                                  block_size: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, block_size)
    value = tl.load(input_ptr + row * d1 + cols, mask=cols < d1, other=0.0)
    tl.store(output_ptr + row, tl.sum(value, axis=0))


@triton.jit
def sum_reduce_rank3_middle_kernel(input_ptr, output_ptr, d1: tl.constexpr,
                                   d2: tl.constexpr,
                                   block_size: tl.constexpr):
    i0 = tl.program_id(0)
    cols = tl.program_id(1) * block_size + tl.arange(0, block_size)
    result = tl.zeros((block_size,), tl.float32)
    for i1 in range(0, d1):
        result += tl.load(
            input_ptr + (i0 * d1 + i1) * d2 + cols,
            mask=cols < d2,
            other=0.0,
        )
    tl.store(output_ptr + i0 * d2 + cols, result, mask=cols < d2)


@triton.jit
def sum_reduce_rank3_inner_kernel(input_ptr, output_ptr, d2: tl.constexpr,
                                  block_size: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, block_size)
    value = tl.load(input_ptr + row * d2 + cols, mask=cols < d2, other=0.0)
    tl.store(output_ptr + row, tl.sum(value, axis=0))


@triton.jit
def sum_reduce_rank3_tail_kernel(input_ptr, output_ptr, d1: tl.constexpr,
                                 d2: tl.constexpr,
                                 block_size: tl.constexpr):
    i0 = tl.program_id(0)
    cols = tl.arange(0, block_size)
    result = 0.0
    for i1 in range(0, d1):
        row = tl.load(
            input_ptr + (i0 * d1 + i1) * d2 + cols,
            mask=cols < d2,
            other=0.0,
        )
        result += tl.sum(row, axis=0)
    tl.store(output_ptr + i0, result)


@triton.jit
def sum_reduce_rank4_noncontiguous_kernel(input_ptr, output_ptr,
                                           d0: tl.constexpr, d1: tl.constexpr,
                                           d2: tl.constexpr, d3: tl.constexpr,
                                           block_size: tl.constexpr):
    i1 = tl.program_id(0)
    cols = tl.program_id(1) * block_size + tl.arange(0, block_size)
    result = tl.zeros((block_size,), tl.float32)
    for i0 in range(0, d0):
        for i2 in range(0, d2):
            result += tl.load(
                input_ptr + ((i0 * d1 + i1) * d2 + i2) * d3 + cols,
                mask=cols < d3,
                other=0.0,
            )
    tl.store(output_ptr + i1 * d3 + cols, result, mask=cols < d3)


def torch_sum(value, axes):
    for axis in sorted(axes, reverse=True):
        value = torch.sum(value, dim=axis)
    return value


def next_power_of_2(value):
    return 1 << (value - 1).bit_length()


def run_sum_reduce(value, axes):
    shape = tuple(value.shape)
    expected = torch_sum(value, axes)
    output = torch.empty_like(expected)

    if len(shape) == 1:
        sum_reduce_rank1_kernel[(1,)](
            value, output, shape[0], next_power_of_2(shape[0])
        )
    elif len(shape) == 2:
        d0, d1 = shape
        block_size = next_power_of_2(d1)
        if axes == (0,):
            sum_reduce_rank2_outer_kernel[(triton.cdiv(d1, block_size),)](
                value, output, d0, d1, block_size
            )
        else:
            sum_reduce_rank2_inner_kernel[(d0,)](
                value, output, d1, block_size
            )
    elif len(shape) == 3:
        d0, d1, d2 = shape
        block_size = next_power_of_2(d2)
        if axes == (1,):
            sum_reduce_rank3_middle_kernel[(d0, triton.cdiv(d2, block_size))](
                value, output, d1, d2, block_size
            )
        elif axes == (2,):
            sum_reduce_rank3_inner_kernel[(d0 * d1,)](
                value, output, d2, block_size
            )
        else:
            sum_reduce_rank3_tail_kernel[(d0,)](
                value, output, d1, d2, block_size
            )
    else:
        d0, d1, d2, d3 = shape
        if axes == (0, 2):
            block_size = next_power_of_2(d3)
            sum_reduce_rank4_noncontiguous_kernel[(d1, triton.cdiv(d3, block_size))](
                value, output, d0, d1, d2, d3, block_size
            )
        else:
            # A staged reduction is the usual Triton formulation for an
            # all-elements reduction that is larger than one practical tile.
            block_size = 1024
            numel = math.prod(shape)
            num_partials = triton.cdiv(numel, block_size)
            partials = torch.empty((num_partials,), device=value.device,
                                   dtype=value.dtype)
            sum_reduce_blocks_kernel[(num_partials,)](
                value, partials, numel, block_size
            )
            sum_reduce_rank1_kernel[(1,)](
                partials, output, num_partials, next_power_of_2(num_partials)
            )
    return output, expected


@pytest.mark.parametrize(
    "shape, axes, _",
    TRITON_CASES,
)
def test_sum_reduce_triton(shape, axes, _, device):
    torch.manual_seed(0)
    value = torch.rand(shape, device=device, dtype=torch.float32)
    output, expected = run_sum_reduce(value, axes)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)
