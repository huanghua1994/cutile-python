# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import enum
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Set, Optional, Mapping, Any, TypeVar, Generic

from cuda.tile._exception import Loc, TileSyntaxError
from cuda.tile._ir.ir import Operation, Var, IRContext


@dataclass
class JumpInfo:
    jump_op: Operation | None
    outputs: tuple[Var, ...]


@dataclass
class ControlFlowInfo:
    stored_names: tuple[str, ...]
    flatten: bool = False
    jumps: list[JumpInfo] = dataclasses.field(default_factory=list)


class LocalScope:
    def __init__(self,
                 all_locals: Set[str],
                 ir_ctx: IRContext,
                 parent: Optional["LocalScope"] = None):
        self._all_locals = all_locals
        self._ir_ctx = ir_ctx
        self._map = dict()
        self._parent = parent

    def is_local_name(self, name: str):
        current = self
        while current is not None:
            if name in current._all_locals:
                return True
            current = current._parent
        return False

    def redefine(self, name: str, loc: Loc) -> Var:
        var = self._ir_ctx.make_var(name, loc)
        self._map[name] = var
        return var

    def __getitem__(self, name: str):
        var = self._lookup(name)
        if var is None:
            raise TileSyntaxError(f"Undefined variable {name} used")
        return var

    def get(self, name: str, loc: Loc):
        var = self._lookup(name)
        if var is None:
            return self._ir_ctx.make_var(name, loc, undefined=True)
        else:
            return var

    def _lookup(self, name: str) -> Optional[Var]:
        seen = set()
        current = self
        while current is not None:
            var = current._map.get(name)
            if var is not None:
                return var
            # Sanity check, should not reach here.
            if id(current) in seen:
                raise RuntimeError("Cycle detected in Scope chain")
            seen.add(id(current))
            current = current._parent
        return None

    @contextmanager
    def enter_branch(self):
        old = self._map
        self._map = _OverlayDict(old)
        try:
            yield
        finally:
            self._map = old


class _OverlayDict:
    def __init__(self, orig_dict: dict):
        self._orig = orig_dict
        self._overlay = dict()

    def get(self, key):
        value = self._overlay.get(key)
        return self._orig.get(key) if value is None else value

    def __setitem__(self, key, value):
        self._overlay[key] = value


class _CurrentScope(threading.local):
    scope = None


_current_scope = _CurrentScope()


class _MissingItem(enum.IntEnum):
    INSTANCE = 0


V = TypeVar("V")


class IntMap(Generic[V]):
    def __init__(self):
        self._items = []

    def __getitem__(self, idx: int):
        assert isinstance(idx, int)
        assert idx >= 0
        try:
            val = self._items[idx]
        except IndexError:
            raise KeyError()
        if val is _MissingItem:
            raise KeyError()
        return val

    def __setitem__(self, idx, value):
        assert isinstance(idx, int)
        assert idx >= 0
        size = len(self._items)
        if idx < size:
            self._items[idx] = value
        else:
            if idx > size:
                self._items.extend((_MissingItem.INSTANCE,) * (idx - size))
            self._items.append(value)


@dataclass
class Scope:
    local: LocalScope
    loop_info: ControlFlowInfo | None
    if_else_info: ControlFlowInfo | None
    frozen_globals: Mapping[str, Any]
    call_site: Loc | None
    hir2ir_varmap: IntMap[Var]

    @contextmanager
    def make_current(self):
        old = _current_scope.scope
        _current_scope.scope = self
        try:
            yield
        finally:
            _current_scope.scope = old

    @staticmethod
    def get_current() -> "Scope | None":
        return _current_scope.scope

    @contextmanager
    def change_loop_info(self, new: ControlFlowInfo):
        old = self.loop_info
        self.loop_info = new
        try:
            yield
        finally:
            self.loop_info = old

    @contextmanager
    def change_if_else_info(self, new: ControlFlowInfo):
        old = self.if_else_info
        try:
            self.if_else_info = new
            yield
        finally:
            self.if_else_info = old
