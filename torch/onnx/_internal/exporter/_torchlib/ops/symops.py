"""Implementation for torch.sym* ops."""

# mypy: disable-error-code="misc,arg-type,type-arg,valid-type,assignment,return-value,type-var,operator,no-untyped-def,index"
# ruff: noqa: TCH001,TCH002
# flake8: noqa

from __future__ import annotations

from onnxscript.onnx_opset import opset18 as op

import torch
from torch.onnx._internal.exporter._torchlib._tensor_typing import BOOL
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl


@onnx_impl(torch.sym_not, trace_only=True)
def sym_not(self: BOOL) -> BOOL:
    """sym_not(SymBool self) -> SymBool"""
    return op.Not(self)
