# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from torch.testing import make_tensor
from typing import Optional, Tuple
from conftest import float_dtypes, int_dtypes, dtype_id
from math import ceil
from util import filecheck, assert_equal, get_bytecode
import cuda.tile as ct
from cuda.tile._exception import TileTypeError
from cuda.tile._numeric_semantics import RoundingMode as RMd


def _squeezed_zeros_like(x, axis: Optional[int | Tuple[int, ...]], keepdims: bool):
    shape = x.shape
    if axis is None:
        squeezed_shape = (1,) * (len(shape) if keepdims else 0)
    else:
        # Normalize to tuple
        if isinstance(axis, int):
            axis = (axis,)
        axis = tuple(a % x.ndim for a in axis)  # handle negative axes
        axis_set = set(axis)
        squeezed_shape = []
        for i, dim in enumerate(shape):
            if i in axis_set:
                if keepdims:
                    squeezed_shape.append(1)
            else:
                squeezed_shape.append(dim)
    return torch.zeros(squeezed_shape, dtype=x.dtype, device="cuda")


def make_reduce_axis1_2d(reduce_op):
    @ct.kernel
    def kernel(
        input, output, B: ct.Constant[int], N: ct.Constant[int], keepdims: ct.Constant[bool]
    ):
        px = ct.bid(0)
        rows = ct.load(input, index=(px, 0), shape=(B, N))
        out = reduce_op(rows, axis=1, keepdims=keepdims)
        if keepdims:
            ct.store(output, index=(px, 0), tile=out)
        else:
            ct.store(output, index=(px, ), tile=out)
    return kernel


def make_reduce_axis1_3d(reduce_op):
    @ct.kernel
    def kernel(
        input, output,
        B: ct.Constant[int], N: ct.Constant[int], M: ct.Constant[int], keepdims: ct.Constant[bool]
    ):
        px = ct.bid(0)
        rows = ct.load(input, index=(px, 0, 0), shape=(B, N, M))
        out = reduce_op(rows, axis=1, keepdims=keepdims)
        if keepdims:
            ct.store(output, index=(px, 0, 0), tile=out)
        else:
            ct.store(output, index=(px, 0), tile=out)
    return kernel


maxmin_cases = [
    pytest.param(ct.max, torch.amax, id="max"),
    pytest.param(ct.min, torch.amin, id="min"),
]


@pytest.mark.parametrize("shape", [(512, 128)])
@pytest.mark.parametrize("tile", [16])
@pytest.mark.parametrize("dtype", float_dtypes, ids=dtype_id)
@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("reduce_op, torch_op", maxmin_cases)
def test_reduce_maxminf(shape, tile, dtype, keepdims, reduce_op, torch_op):
    x = torch.rand(shape, dtype=dtype, device="cuda") * 2 - 1
    y = _squeezed_zeros_like(x, axis=1, keepdims=keepdims)
    grid = (ceil(shape[0] / tile), 1, 1)
    kernel = make_reduce_axis1_2d(reduce_op)
    ct.launch(torch.cuda.current_stream(), grid, kernel, (x, y, tile, shape[1], keepdims))
    ref_result = torch_op(x, dim=1, keepdim=keepdims)
    torch.testing.assert_close(y, ref_result)


@pytest.mark.parametrize("shape", [(512, 128)])
@pytest.mark.parametrize("tile", [16])
@pytest.mark.parametrize("dtype", int_dtypes, ids=dtype_id)
@pytest.mark.parametrize("low", [-100])
@pytest.mark.parametrize("high", [-20, 100])
@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("reduce_op, torch_op", maxmin_cases)
def test_reduce_maxmini(shape, tile, dtype, low, high, keepdims, reduce_op, torch_op):
    x = torch.randint(low, high + 1, shape, dtype=dtype, device="cuda")
    y = _squeezed_zeros_like(x, axis=1, keepdims=keepdims)
    grid = (ceil(shape[0] / tile), 1, 1)
    kernel = make_reduce_axis1_2d(reduce_op)
    ct.launch(torch.cuda.current_stream(), grid, kernel, (x, y, tile, shape[1], keepdims))
    ref_result = torch_op(x, dim=1, keepdim=keepdims)
    torch.testing.assert_close(y, ref_result)
    torch.testing.assert_close(y, ref_result)


def make_reduce_axisNone(reduce_op):
    @ct.kernel
    def kernel(input, output,
               B: ct.Constant[int],
               N: ct.Constant[int],
               keepdims: ct.Constant[bool]):
        rows = ct.load(input, index=(0, 0), shape=(B, N))
        if keepdims:
            out = reduce_op(rows, axis=None, keepdims=keepdims)
        else:
            out = reduce_op(rows, axis=None, keepdims=keepdims)
            out = ct.full((1, 1), out.item(), dtype=out.dtype)
        ct.store(output, index=(0, 0), tile=out)
    return kernel


@pytest.mark.parametrize("shape", [(512, 128)])
@pytest.mark.parametrize("dtype", float_dtypes, ids=dtype_id)
@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("reduce_op, torch_op", maxmin_cases)
def test_reduce_maxminf_all_axes(shape, dtype, keepdims, reduce_op, torch_op):
    x = torch.rand(shape, dtype=dtype, device="cuda") * 2 - 1
    grid = (1, 1, 1)
    kernel = make_reduce_axisNone(reduce_op)
    if keepdims:
        y = _squeezed_zeros_like(x, axis=None, keepdims=keepdims)
        ct.launch(torch.cuda.current_stream(), grid, kernel, (x, y, shape[0], shape[1], keepdims))
    else:
        y = torch.zeros((1,) * len(shape), dtype=dtype, device="cuda")
        ct.launch(torch.cuda.current_stream(), grid, kernel, (x, y, shape[0], shape[1], keepdims))
        y = y.squeeze()
    ref_result = torch_op(x, dim=None, keepdim=keepdims)
    torch.testing.assert_close(y, ref_result)


def make_reduce_max_two_axes(reduce_op):
    @ct.kernel
    def kernel(input, output,
               TILE: ct.Constant[int],
               N: ct.Constant[int],
               M: ct.Constant[int],
               keepdims: ct.Constant[bool],
               a1: ct.Constant[int],
               a2: ct.Constant[int]):
        px = ct.bid(0)
        rows = ct.load(input, index=(px, 0, 0), shape=(TILE, N, M))
        out = reduce_op(rows, axis=(a1, a2), keepdims=keepdims)
        if keepdims:
            ct.store(output, index=(px, 0, 0), tile=out)
        else:
            ct.store(output, index=(px, ), tile=out)
    return kernel


@pytest.mark.parametrize("shape", [(32, 32, 64)])
@pytest.mark.parametrize("tile", [16])
@pytest.mark.parametrize("dtype", float_dtypes, ids=dtype_id)
@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("reduce_op, torch_op", maxmin_cases)
@pytest.mark.parametrize("axes", [(1, 2), (2, 1), (-1, -2), (-2, -1), (-2, 2), (2, -2)])
def test_reduce_maxminf_two_axes(shape, tile, dtype, keepdims, reduce_op, torch_op, axes):
    x = torch.rand(shape, dtype=dtype, device="cuda") * 2 - 1
    y = _squeezed_zeros_like(x, axis=(1, 2), keepdims=keepdims)
    grid = (ceil(shape[0] / tile), 1, 1)
    kernel = make_reduce_max_two_axes(reduce_op)
    ct.launch(torch.cuda.current_stream(), grid, kernel,
              (x, y, tile, shape[1], shape[2], keepdims, *axes))
    ref_result = torch_op(x, dim=(1, 2), keepdim=keepdims)
    torch.testing.assert_close(y, ref_result)


def test_reduce_repeated_axis_error():
    @ct.kernel
    def kernel(x):
        tx = ct.load(x, (0, 0), (16, 16))
        ct.sum(tx, axis=(1, 1))

    x = torch.rand((16, 16), dtype=torch.float32, device="cuda")
    with pytest.raises(TileTypeError, match="Repeated reduction axis 1"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))


sumprod_cases = [
    pytest.param(ct.sum, torch.sum, id="sum"),
    pytest.param(ct.prod, torch.prod, id="prod"),
]


@pytest.mark.parametrize("shape", [(512, 128)])
@pytest.mark.parametrize("tile", [16])
@pytest.mark.parametrize("dtype", float_dtypes, ids=dtype_id)
@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("reduce_op, torch_op", sumprod_cases)
def test_reduce_sumprodf(shape, tile, dtype, keepdims, reduce_op, torch_op):
    if reduce_op is ct.sum and (dtype is torch.bfloat16 or dtype is torch.float16):
        pytest.xfail("Bf16/Fp16 reduce sum introduce a difference from torch.")

    x = torch.rand(shape, dtype=dtype, device="cuda") * 2 - 1
    y = _squeezed_zeros_like(x, axis=1, keepdims=keepdims)
    grid = (ceil(shape[0] / tile), 1, 1)
    kernel = make_reduce_axis1_2d(reduce_op)
    ct.launch(torch.cuda.current_stream(), grid, kernel, (x, y, tile, shape[1], keepdims))
    ref_result = torch_op(x, dim=1, keepdim=keepdims).to(dtype)
    torch.testing.assert_close(y, ref_result)


@pytest.mark.parametrize("shape", [(512, 128)])
@pytest.mark.parametrize("tile", [16])
@pytest.mark.parametrize("dtype", int_dtypes, ids=dtype_id)
@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("reduce_op, torch_op", sumprod_cases)
def test_reduce_sumprodi(shape, tile, dtype, keepdims, reduce_op, torch_op):
    x = torch.randint(-100, 100, shape, dtype=dtype, device="cuda")
    y = _squeezed_zeros_like(x, axis=1, keepdims=keepdims)
    grid = (ceil(shape[0] / tile), 1, 1)
    kernel = make_reduce_axis1_2d(reduce_op)
    ct.launch(torch.cuda.current_stream(), grid, kernel, (x, y, tile, shape[1], keepdims))
    ref_result = torch_op(x, dim=1, keepdim=keepdims).to(dtype)
    torch.testing.assert_close(y, ref_result)


@pytest.mark.parametrize("shape", [(512, 128)])
@pytest.mark.parametrize("tile", [16])
@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("reduce_op, torch_op", sumprod_cases)
def test_reduce_sumprodb(shape, tile, keepdims, reduce_op, torch_op):
    x = torch.randint(0, 2, shape, dtype=torch.bool, device="cuda")
    y = _squeezed_zeros_like(x, axis=1, keepdims=keepdims).to(torch.int32)
    grid = (ceil(shape[0] / tile), 1, 1)
    kernel = make_reduce_axis1_2d(reduce_op)
    ct.launch(torch.cuda.current_stream(), grid, kernel, (x, y, tile, shape[1], keepdims))
    ref_result = torch_op(x, dim=1, keepdim=keepdims).to(torch.int32)
    torch.testing.assert_close(y, ref_result)


@pytest.mark.parametrize("shape", [(512, 128)])
@pytest.mark.parametrize("dtype", float_dtypes, ids=dtype_id)
@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("reduce_op, torch_op", sumprod_cases)
def test_reduce_sumprodf_all_axes(shape, dtype, keepdims, reduce_op, torch_op):
    x = torch.rand(shape, dtype=dtype, device="cuda")
    grid = (1, 1, 1)
    kernel = make_reduce_axisNone(reduce_op)
    if keepdims:
        y = _squeezed_zeros_like(x, axis=None, keepdims=True)
        ct.launch(torch.cuda.current_stream(), grid, kernel, (x, y, shape[0], shape[1], keepdims))
    else:
        y = torch.zeros((1,) * len(shape), dtype=dtype, device="cuda")
        ct.launch(torch.cuda.current_stream(), grid, kernel, (x, y, shape[0], shape[1], keepdims))
        y = y.squeeze()
    if torch_op is torch.sum:
        ref_result = torch_op(x, dim=None, keepdim=keepdims)
        # Sum can be unstable, so we compare the average.
        atol, rtol = (1e-5, 1e-6) if dtype is torch.float32 else (1e-5, 1e-2)
        torch.testing.assert_close(y / x.numel(), ref_result / x.numel(), atol=atol, rtol=rtol)
    else:
        ref_result = torch_op(x)
        if keepdims:
            ref_result = ref_result.reshape([1] * x.ndim)
        torch.testing.assert_close(y, ref_result)


def make_sumprod_rounding_mode(reduce_op, rounding_mode):
    @ct.kernel
    def kernel(input, output, B: ct.Constant[int], N: ct.Constant[int]):
        px = ct.bid(0)
        rows = ct.load(input, index=(px, 0), shape=(B, N))
        out = reduce_op(rows, axis=1, keepdims=True, rounding_mode=rounding_mode)
        ct.store(output, index=(px, 0), tile=out)
    return kernel


@pytest.mark.use_mlir
@pytest.mark.parametrize("shape", [(512, 128)])
@pytest.mark.parametrize("tile", [16])
@pytest.mark.parametrize("dtype", float_dtypes, ids=dtype_id)
@pytest.mark.parametrize("op_func, tile_op", [(ct.sum, "addf"), (ct.prod, "mulf")])
@pytest.mark.parametrize("rounding_mode",
                         [RMd.RN, RMd.RZ, RMd.RM, RMd.RP, RMd.FULL, RMd.APPROX, RMd.RZI])
def test_reduce_sumprodf_rounding_mode(
    shape, tile, dtype, op_func, tile_op, rounding_mode
):
    should_raise_rounding_mode = rounding_mode in [RMd.RZI, RMd.APPROX, RMd.FULL]
    x = make_tensor(shape, dtype=dtype, device='cuda')
    y = _squeezed_zeros_like(x, axis=1, keepdims=True)
    grid = (ceil(shape[0] / tile), 1, 1)
    kernel = make_sumprod_rounding_mode(op_func, rounding_mode)
    if should_raise_rounding_mode:
        with pytest.raises(TileTypeError,
                           match=fr"Rounding mode {rounding_mode.value} is not supported"):
            ct.launch(torch.cuda.current_stream(), grid, kernel, (x, y, tile, shape[1]))
    else:
        bytecode = get_bytecode(kernel, (x, y, tile, shape[1]))
        if rounding_mode is RMd.RN:
            # Rmd.RN as the default rounding mode is not included in the mlir text
            check_directive = "// CHECK-NOT: rounding<{{[^>]*}}>"
        else:
            check_directive = (
                f"// CHECK: %[[RES:.*]] = {tile_op} %[[A:.*]] rounding<{rounding_mode.value}>"
            )
        filecheck(bytecode, check_directive)
        ct.launch(torch.cuda.current_stream(), grid, kernel, (x, y, tile, shape[1]))


def make_reduce_flush_to_zero(reduce_op, flush_to_zero):
    @ct.kernel
    def kernel(input, output, B: ct.Constant[int], N: ct.Constant[int]):
        px = ct.bid(0)
        rows = ct.load(input, index=(px, 0), shape=(B, N))
        out = reduce_op(rows, axis=1, keepdims=True, flush_to_zero=flush_to_zero)
        ct.store(output, index=(px, 0), tile=out)
    return kernel


@pytest.mark.use_mlir
@pytest.mark.parametrize("shape", [(512, 128)])
@pytest.mark.parametrize("tile", [16])
@pytest.mark.parametrize("dtype", float_dtypes, ids=dtype_id)
@pytest.mark.parametrize("reduce_op, tile_op",
                         [(ct.max, "maxf"), (ct.min, "minf"), (ct.sum, "addf"), (ct.prod, "mulf")])
@pytest.mark.parametrize("flush_to_zero", [True, False])
def test_reduce_flush_to_zero(shape, tile, dtype, reduce_op, tile_op, flush_to_zero):
    should_raise = flush_to_zero and (dtype != torch.float32)
    x = make_tensor(shape, dtype=dtype, device='cuda')
    y = _squeezed_zeros_like(x, axis=1, keepdims=True)
    grid = (ceil(shape[0] / tile), 1, 1)
    kernel = make_reduce_flush_to_zero(reduce_op, flush_to_zero)
    if should_raise:
        with pytest.raises(TileTypeError,
                           match=r"Flush to zero can only be used for float32 type"):
            ct.launch(torch.cuda.current_stream(), grid, kernel, (x, y, tile, shape[1]))
    else:
        bytecode = get_bytecode(kernel, (x, y, tile, shape[1]))
        if flush_to_zero:
            check_directive = f"// CHECK: %[[RES:.*]] = {tile_op} %[[A:.*]] flush_to_zero :"
        else:
            check_directive = f"// CHECK: %[[RES:.*]] = {tile_op} %[[A:.*]]{{{{[[:space:]]*}}}}:"
        filecheck(bytecode, check_directive)
        ct.launch(torch.cuda.current_stream(), grid, kernel, (x, y, tile, shape[1]))


argmaxmin_cases = [
    pytest.param(ct.argmax, torch.argmax, id="argmax"),
    pytest.param(ct.argmin, torch.argmin, id="argmin"),
]


@pytest.mark.parametrize("shape", [(32, 16), (2, 4, 4)])
@pytest.mark.parametrize("tile", [16])
@pytest.mark.parametrize("dtype", float_dtypes+int_dtypes, ids=dtype_id)
@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("reduce_op, torch_op", argmaxmin_cases)
def test_reduce_argmaxmin(shape, tile, dtype, keepdims, reduce_op, torch_op):
    x = make_tensor(shape, dtype=dtype, device='cuda')
    y = _squeezed_zeros_like(x, axis=1, keepdims=keepdims).to(torch.int32)
    grid = (ceil(shape[0] / tile), 1, 1)
    if len(shape) == 2:
        kernel = make_reduce_axis1_2d(reduce_op)
        args = (x, y, tile, shape[1], keepdims)
    else:
        kernel = make_reduce_axis1_3d(reduce_op)
        args = (x, y, tile, shape[1], shape[2], keepdims)
    ct.launch(torch.cuda.current_stream(), grid, kernel, args)
    ref_result = torch_op(x, dim=1, keepdim=keepdims).to(torch.int32)
    assert_equal(y, ref_result)


@pytest.mark.parametrize("shape", [(512, 128)])
@pytest.mark.parametrize("dtype", float_dtypes + int_dtypes, ids=dtype_id)
@pytest.mark.parametrize("reduce_op, torch_op", argmaxmin_cases)
@pytest.mark.parametrize("keepdims", [True, False])
def test_reduce_argmaxmin_all_axes(shape, dtype, reduce_op, torch_op, keepdims):
    x = make_tensor(shape, dtype=dtype, device='cuda')
    grid = (1, 1, 1)
    kernel = make_reduce_axisNone(reduce_op)
    if keepdims:
        y = _squeezed_zeros_like(x, axis=None, keepdims=keepdims).to(torch.int32)
        ct.launch(torch.cuda.current_stream(), grid, kernel, (x, y, shape[0], shape[1], keepdims))
    else:
        y = torch.zeros((1,) * len(shape), dtype=dtype, device="cuda").to(torch.int32)
        ct.launch(torch.cuda.current_stream(), grid, kernel, (x, y, shape[0], shape[1], keepdims))
        y = y.squeeze()
    ref_result = torch_op(x, dim=None, keepdim=keepdims).to(torch.int32)
    assert_equal(y, ref_result)
