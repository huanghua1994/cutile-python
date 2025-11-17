# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import enum
import inspect
import sys
from collections import defaultdict
from dataclasses import dataclass
from types import FunctionType
from typing import Tuple, Any, Optional, Sequence, Callable

from cuda.tile._debug import CUDA_TILE_LOGS
from cuda.tile._exception import (
    TileTypeError,
    TileInternalError, ConstFoldNotImplementedError,
    ConstantNotFoundError, TileSyntaxError, Loc, TileError
)
from cuda.tile._ir import ir
from cuda.tile._ir.ir import Operation, Function, Block, Var, Argument, IRContext, TypedOperation
from cuda.tile._ir.op_impl import op_implementations
from cuda.tile._ir.typing_support import typeof_pyval, get_signature
from cuda.tile._ir.type import FunctionTy, BoundMethodTy, DTypeConstructor, Type, UNDEFINED


class ConstantState(enum.Enum):
    UNSET = 0
    MAY_BE_CONSTANT = 1
    NONCONSTANT = 2


@dataclass
class PhiState:
    constant_state: ConstantState = ConstantState.UNSET
    constant_value: Any = None

    def set_nonconstant(self):
        self.constant_state = ConstantState.NONCONSTANT

    def set_branch_constant(self, value: Any):
        if self.constant_state == ConstantState.UNSET:
            self.constant_state = ConstantState.MAY_BE_CONSTANT
            self.constant_value = value
        elif self.constant_state == ConstantState.MAY_BE_CONSTANT and value != self.constant_value:
            self.constant_state = ConstantState.NONCONSTANT


class TypingContext:
    def __init__(self, ir_ctx: IRContext) -> None:
        self.ir_ctx = ir_ctx
        self.phis = defaultdict(PhiState)

    @property
    def typemap(self):
        return self.ir_ctx.typemap

    @property
    def constants(self):
        return self.ir_ctx.constants

    @property
    def range_infos(self):
        return self.ir_ctx.range_infos

    def get_constant(self, var: Var) -> Any:
        if var.name in self.constants:
            return self.constants[var.name]
        raise ConstantNotFoundError(var.name)

    def try_get_constant(self, var: Var) -> Optional[Any]:
        if var.name in self.constants:
            return self.constants[var.name]
        return None

    def is_constant(self, var: Var) -> bool:
        return var.name in self.constants

    def set_constant(self, var: Var, value: Any):
        if var.name in self.constants:
            raise KeyError(f'Attempt to overwrite existing constant of variable {var.name}')
        self.constants[var.name] = value

    def get_type(self, var: Var) -> Type:
        if var.is_undefined():
            return UNDEFINED
        if var.name in self.typemap:
            return self.typemap[var.name]
        raise KeyError(f"Type for {var.name} not found")

    def set_type(self, var: Var, typ: Type) -> None:
        self.typemap[var.name] = typ

    def phi_propagate_constant(self, src: Var, dst: Var):
        phi = self.phis[dst.name]
        if self.is_constant(src):
            phi.set_branch_constant(self.get_constant(src))
        else:
            phi.set_nonconstant()

    def phi_finalize_constant(self, dst: Var):
        phi = self.phis[dst.name]
        if phi.constant_state == ConstantState.MAY_BE_CONSTANT:
            self.set_constant(dst, phi.constant_value)


def propagate_type(context: TypingContext, name: str, typ: Type) -> None:
    # Propagate the type of the variable to the destination variable.
    # Undefined types are propagated to the destination variable.
    if name not in context.typemap or typ is UNDEFINED:
        context.typemap[name] = typ
    elif context.typemap[name] is UNDEFINED:
        pass
    elif context.typemap[name] != typ:
        # TODO: better error message to show the variable location.
        raise TypeError(f"Types mismatch for {name} in propagation: "
                        f"{context.typemap[name]} != {typ}")


def infer_constant(op: Operation, context: TypingContext) -> bool:
    const_results = None
    try:
        const_results = op.fold_constant(typing_context=context)
    except (ConstFoldNotImplementedError, ConstantNotFoundError):
        pass
    except Exception as e:
        raise TileTypeError(str(e))
    else:
        if len(op.result_vars) == 1:
            const_results = [const_results]
        if len(const_results) != len(op.result_vars):
            raise TileInternalError(f"Number of results mismatch: "
                                    f"{len(const_results)} != {len(op.result_vars)}", op.loc)
        for res_var, res_val in zip(op.result_vars, const_results):
            try:
                # TODO: we are using typeof_pyval to get the type of the folded constant
                # which may not be enough when the fold_constant
                # computes a np.float64 and pyval can treat it as a default float type
                res_type = typeof_pyval(res_val)
            except TypeError as e:
                raise TileTypeError(str(e))
            context.set_type(res_var, res_type)
            context.set_constant(res_var, res_val)

    return const_results is not None


def infer_type(op: Operation, context: TypingContext) -> None:
    try:
        res_types = op.infer_type(context)
    except TypeError as e:
        raise TileTypeError(str(e), op.loc) from e
    if isinstance(res_types, Type):
        res_types = [res_types]
    if len(res_types) != len(op.result_vars):
        raise TileInternalError(f"Number of results mismatch: "
                                f"{len(res_types)} != {len(op.result_vars)}", op.loc)
    for result_var, res_type in zip(op.result_vars, res_types):
        context.set_type(result_var, res_type)


def _flatten_if_else(block: Block, idx: int, context: TypingContext):
    from cuda.tile._ir.ops import EndBranch, Assign, Continue, Break
    op = block[idx]
    branch_taken = op.then_block if context.constants[op.cond.name] else op.else_block
    old_ops = branch_taken.detach_all()
    new_ops = []
    early_stop = False
    for inner_op in old_ops:
        if isinstance(inner_op, EndBranch):
            for result_var, var in zip(op.result_vars, inner_op.outputs):
                new_ops.append(Assign(var, result_var, op.loc))
        else:
            if isinstance(inner_op, (Continue, Break)):
                early_stop = True
            new_ops.append(inner_op)
    if early_stop:
        del block[idx+1:]
    block[idx:idx+1] = new_ops


def _flatten_loop(block: Block, idx: int) -> bool:
    from cuda.tile._ir.ops import Loop, Break, Assign

    loop = block[idx]
    if (not isinstance(loop, Loop)
            or loop.for_loop is not None
            or not isinstance(loop.body[-1], Break)
            or _have_break_or_continue(loop.body[:-1])):
        return False

    new_ops = []
    for init_var, body_var in zip(loop.carried_vars.initial, loop.carried_vars.body, strict=True):
        new_ops.append(Assign(init_var, body_var, loop.loc))

    *body_ops, brek = loop.body.detach_all()
    new_ops.extend(body_ops)

    for break_res, loop_res in zip(brek.output_vars, loop.carried_vars.results, strict=True):
        new_ops.append(Assign(break_res, loop_res, brek.loc))
    block[idx:idx+1] = new_ops
    return True


def _have_break_or_continue(ops):
    from cuda.tile._ir.ops import Loop, Break, Continue
    return any(
        isinstance(op, (Break, Continue))
        or (not isinstance(op, Loop)
            and any(_have_break_or_continue(block.operations) for block in op.nested_blocks))
        for op in ops
    )


def _add_constant(value, block: Block, loc: Loc, var: Var, ctx: TypingContext, dtype=None):
    from cuda.tile._ir.ops import const
    const(value, block, loc, var, dtype=dtype)
    ctx.set_constant(var, value)
    ctx.set_type(var, dtype or typeof_pyval(value))


def _bind_args(sig_func, args, kwargs, block, loc, ctx) -> Sequence[Var]:
    sig = get_signature(sig_func)
    try:
        bound_args = sig.bind(*args, **kwargs)
    except TypeError as e:
        raise TileTypeError(f"{sig_func.__name__}(): {e}", loc)
    ret = []
    for name, param in sig.parameters.items():
        if name in bound_args.arguments:
            ret.append(bound_args.arguments[name])
        elif param.kind == param.VAR_POSITIONAL:
            ret.append(())
        else:
            assert param.default is not param.empty
            var = block.make_temp_var(loc)
            _add_constant(param.default, block, loc, var, ctx)
            ret.append(var)
    return ret


def _check_recursive_call(call_loc: Loc, callee: Callable):
    while call_loc is not None:
        if call_loc.function is callee:
            raise TileTypeError("Recursive function call detected")
        call_loc = call_loc.call_site


def _replace_call(block: Block, idx: int, ctx: TypingContext):
    from cuda.tile._ir.ops import GetBoundSelf, assign, typed_const

    op = block[idx]
    new_block = Block(block.ctx)
    ty = ctx.get_type(op.func)

    args = []
    if isinstance(ty, FunctionTy):
        callee = ty.func
    elif isinstance(ty, BoundMethodTy):
        callee = ty.func
        self_var = new_block.make_temp_var(op.loc)
        new_block.append(GetBoundSelf(op.func, self_var, op.loc))
        ctx.set_type(self_var, ty.self_ty)
        args.append(self_var)
    elif isinstance(ty, DTypeConstructor):
        callee = ty.dtype
    else:
        raise TileTypeError(f"Cannot call an object of type {ty}")

    args.extend(op.args)
    arg_list = _bind_args(callee, args, op.kwarg_dict(), new_block, op.loc, ctx)

    if callee in op_implementations:
        with ir.Builder(ctx.ir_ctx, op.loc) as ir_builder:
            result = op_implementations[callee](*arg_list)
            if result is None:
                result = typed_const(None)

        assert isinstance(result, Var)

        mapper = ir.Mapper(ctx.ir_ctx, preserve_vars=True)
        mapper.set_var(result, op.result_var)
        ctx.ir_ctx.copy_type_information(result, op.result_var)

        for new_op in ir_builder.ops:
            new_block.append(new_op.clone(mapper))

        # If the returned result variable is not produced by any of the newly created operations,
        # insert an Assign op.
        #
        # This mainly happens when an operation implementation reduces to a no-op by returning
        # its input. For example, reshape(x, new_shape) may return x when the new shape is the same
        # as the old one. So we need to replace `y = reshape(x, new_shape)` with `y = assign(x)`
        # to make sure `y` is still defined.
        if not any(result.name == r.name
                   for new_op in ir_builder.ops for r in new_op.result_vars):
            assign(result, new_block, op.loc, op.result_var)
    else:
        # Callee is a user-defined function.
        from cuda.tile._ast2ir import get_function_ir
        _check_recursive_call(op.loc, callee)

        sig = get_signature(callee)
        for param_name, param in sig.parameters.items():
            if param.kind in (inspect.Parameter.VAR_POSITIONAL,
                              inspect.Parameter.VAR_KEYWORD):
                raise TileSyntaxError("Variadic parameters in user-defined"
                                      " functions are not supported")
        callee_function_ir = get_function_ir(callee, new_block.ctx, call_site=op.loc)
        for arg, param in zip(arg_list, callee_function_ir.parameters):
            assign(arg, new_block, op.loc, param)
        new_block.extend(callee_function_ir.root_block.detach_all())
        assign(callee_function_ir.return_value, new_block, op.loc, op.result_var)

    block[idx:idx+1] = new_block.detach_all()


def infer_types_for_op(context: TypingContext, block: Block, i: int) -> int:
    from cuda.tile._ir.ops import IfElse, Const, Call

    op = block[i]

    if _flatten_loop(block, i):
        return 0

    if isinstance(op, IfElse) and op.cond.name in context.constants:
        _flatten_if_else(block, i, context)
        return 0

    if isinstance(op, Call):
        _replace_call(block, i, context)
        return 0

    if isinstance(op, Const):
        context.constants[op.result_var.name] = op.value
        if op.dtype is None:
            try:
                type = typeof_pyval(op.value)
            except TypeError as e:
                raise TileTypeError(str(e), op.loc)
        else:
            type = op.dtype
        context.set_type(op.result_var, type)
    elif infer_constant(op, context):
        if len(op.result_vars) == 1 and \
                op.result_var.name in context.constants and \
                not isinstance(op, Const):
            const_val = context.constants[op.result_var.name]
            block[i] = Const(const_val, op.result_var, op.loc)
    else:
        infer_type(op, context)

    return 1


def infer_types_in_block(context: TypingContext, block: Block) -> None:
    i = 0
    while i < len(block):
        op = block[i]
        if isinstance(op, TypedOperation):
            i += 1
            continue

        with op.loc:
            try:
                i += infer_types_for_op(context, block, i)
            except TileError:
                raise
            except Exception as e:
                raise TileInternalError(str(e)) from e


def infer_types_in_func(context: TypingContext, func: Function, args: Tuple[Argument, ...]) -> None:
    if len(args) != len(func.parameters):
        msg = f"Expected {len(func.parameters)} arguments, got {len(args)}"
        raise TileTypeError(msg, func.loc)

    # Initialize the typemap and const map with input args
    for var, arg in zip(func.parameters, args):
        context.set_type(var, arg.type)
        if arg.is_const:
            context.set_constant(var, arg.const_value)

    infer_types_in_block(context, func.root_block)


def infer_types_pass(func: Function, args: Tuple[Argument, ...], pyfunc: FunctionType):
    context = TypingContext(func.root_block.ctx)
    try:
        infer_types_in_func(context, func, args)
    except Exception as e:
        if 'CUTILEIR' in CUDA_TILE_LOGS:
            highlight_loc = e.loc if hasattr(e, 'loc') else None
            code = (f"====Partial CuTile IR for {func}==== \n\n"
                    f"{func.to_string(highlight_loc=highlight_loc)}\n\n")
            print(f'\n{code}', file=sys.stderr)
        raise
