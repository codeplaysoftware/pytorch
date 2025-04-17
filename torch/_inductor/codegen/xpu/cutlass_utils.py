# mypy: allow-untyped-defs
import functools
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Optional

import sympy

import torch
from torch._inductor.utils import clear_on_fresh_inductor_cache

from ... import config
from ...ir import Layout
from ...runtime.runtime_utils import cache_dir
from ...virtualized import V


log = logging.getLogger(__name__)


@functools.lru_cache(None)
def try_import_cutlass() -> bool:
    """
    Currently only supporting user specified cutlass_dir or falling to the
    default ../third_party/cutlass/ (build from source setups).
    """
    # Copy CUTLASS python scripts to a temp dir and add the temp dir to Python search path.

    cutlass_py_full_path = os.path.abspath(
        os.path.join(config.cutlass_dir, "python/cutlass_library")
    )
    tmp_cutlass_py_full_path = os.path.abspath(
        os.path.join(cache_dir(), "torch_cutlass_library")
    )
    dst_link = os.path.join(tmp_cutlass_py_full_path, "cutlass_library")

    if os.path.isdir(cutlass_py_full_path):
        if tmp_cutlass_py_full_path not in sys.path:
            if os.path.exists(dst_link):
                assert os.path.islink(dst_link), (
                    f"{dst_link} is not a symlink. Try to remove {dst_link} manually and try again."
                )
                assert os.path.realpath(os.readlink(dst_link)) == os.path.realpath(
                    cutlass_py_full_path
                ), f"Symlink at {dst_link} does not point to {cutlass_py_full_path}"
            else:
                os.makedirs(tmp_cutlass_py_full_path, exist_ok=True)
                os.symlink(cutlass_py_full_path, dst_link)
            sys.path.append(tmp_cutlass_py_full_path)
        try:
            import cutlass_library.generator  # noqa: F401
            import cutlass_library.library  # noqa: F401
            import cutlass_library.manifest  # noqa: F401

            return True
        except ImportError as e:
            log.debug(
                "Failed to import CUTLASS packages: %s, ignoring the CUTLASS backend.",
                str(e),
            )
    else:
        log.debug(
            "Failed to import CUTLASS packages: CUTLASS repo does not exist: %s",
            cutlass_py_full_path,
        )
    return False


@functools.lru_cache(8)
def _normalize_sycl_arch(arch: str) -> str:
    if int(arch) == 11:
        return "11"
    else:
        raise NotImplementedError(f"Unsupported sycl arch: {arch}")


@dataclass
class CUTLASSArgs:
    """
    CUTLASS args used to initialize a CUTLASS Manifest.
    """

    architectures: Optional[str] = None
    cuda_version: Optional[str] = None  # Unused in generator.py for PVC
    instantiation_level: Optional[str] = None  # Unused YET in generator.py for PVC

    operations = "all"
    build_dir = ""
    curr_build_dir = ""
    generator_target = ""
    kernels = "all"
    ignore_kernels = ""
    exclude_kernels = ""
    # UNUSED at the moment, part of Manifest class in cutlass_library
    kernel_filter_file: None = None
    selected_kernel_list: None = None
    interface_dir: None = None
    filter_by_cc = False
    disable_full_archs_compilation = False

    def __post_init__(self):
        if self.architectures is None:
            raise RuntimeError(f"{self.architectures=} is None!")
        self.architectures = _normalize_sycl_arch(self.architectures)


@clear_on_fresh_inductor_cache
@functools.lru_cache(None)
def _gen_ops_cached(arch) -> list[Any]:
    # Import cutlass python scripts.
    assert try_import_cutlass()
    import cutlass_library.generator as cutlass_generator
    import cutlass_library.manifest as cutlass_manifest

    if arch is None:
        log.error(
            "Cannot detect XPU arch %s."
            "Will discard all cutlass ops. "
            "Please consider setting _inductor.xpu.arch",
            arch,
        )
        return []
    arch = _normalize_sycl_arch(arch)

    sycl_version = "2025.0.1"  # Placeholder, Unused in GeneratePVC

    args = CUTLASSArgs(
        architectures=arch,
        instantiation_level="0",  # TODO (SYCL) : Make it config param once enabled in cutlass_library/generator.py
        cuda_version=sycl_version,
    )
    manifest = cutlass_manifest.Manifest(args)

    if arch == "11":
        cutlass_generator.GeneratePVC(manifest, sycl_version)
    else:
        log.error("Invalid XPU arch")
        return []
    return manifest.operations


def gen_ops() -> list[Any]:
    """
    Generates all supported CUTLASS operations.
    """
    # Currently limited to PVC (arch 1100), harcoding arch
    # TODO :(SYCL) get_xpu_arch()
    arch = "11"
    return _gen_ops_cached(arch)


def torch_dtype_to_cutlass_type(
    torch_dtype: torch.dtype,
) -> "cutlass_library.library.DataType":  # type: ignore[name-defined] # noqa: F821
    # Import cutlass python scripts.
    assert try_import_cutlass()
    import cutlass_library  # type: ignore[import]

    if torch_dtype == torch.float:
        return cutlass_library.library.DataType.f32
    elif torch_dtype == torch.half:
        return cutlass_library.library.DataType.f16
    elif torch_dtype == torch.bfloat16:
        return cutlass_library.library.DataType.bf16
    else:
        raise NotImplementedError(f"Unsupported data type: {torch_dtype=}")


def dtype_match(
    torch_dtype: Optional[torch.dtype],
    cutlass_dtype: "cutlass_library.library.DataType",  # type: ignore[name-defined]  # noqa: F821
) -> bool:
    # Import cutlass python scripts.
    assert try_import_cutlass()
    import cutlass_library

    if torch_dtype == torch.float:
        return cutlass_dtype == cutlass_library.library.DataType.f32
    elif torch_dtype == torch.half:
        return cutlass_dtype == cutlass_library.library.DataType.f16
    elif torch_dtype == torch.bfloat16:
        return cutlass_dtype == cutlass_library.library.DataType.bf16
    elif torch_dtype == torch.int8:
        return cutlass_dtype == cutlass_library.library.DataType.s8
    elif torch_dtype == torch.uint8:
        return cutlass_dtype == cutlass_library.library.DataType.u8
    elif torch_dtype == torch.int32:
        return cutlass_dtype == cutlass_library.library.DataType.s32
    else:
        return False


def get_accumulator_dtype(
    input_torch_dtypes: list[torch.dtype],
) -> Optional[torch.dtype]:
    """
    Given a pair of input torch dtypes, returns the inferred accumulator torch dtype.
    """
    # TODO (SYCL) : Extend this once other accumulator & input types are supported
    if len(input_torch_dtypes) != 2:
        return None

    if all(dtype == torch.bfloat16 for dtype in input_torch_dtypes):
        return torch.float
    else:
        raise NotImplementedError(f"Unsupported data types: {input_torch_dtypes}")


def get_alignments(torch_dtype: torch.dtype) -> list[int]:
    """
    Returns all possible valid CUTLASS alignments in terms of the number of elements for a given dtype.
    """
    # TODO (SYCL): Extend for other types & double-check alignments
    if torch_dtype == torch.bfloat16:
        return [8, 4, 2, 1]
    elif torch_dtype == torch.float:
        return [4, 2, 1]
    else:
        raise NotImplementedError(f"unsupported {torch_dtype=} for alignments")


def get_max_alignment(inductor_layout: Layout) -> int:
    """
    Returns the max alignment (in terms of number of elements) for a given Inductor Layout.
    """

    dtype = inductor_layout.dtype
    size = inductor_layout.size
    offset = inductor_layout.offset

    def is_static_int(number):
        return isinstance(number, (int, sympy.Integer))

    def a_factor_of(x, alignment):
        if is_static_int(x) and is_static_int(alignment):
            return x % alignment == 0
        rem = sympy.Mod(x, alignment)
        return V.graph.sizevars.evaluate_expr(sympy.Eq(rem, 0))

    try:
        contiguous_dim = inductor_layout.stride.index(1)
    except ValueError:
        # No dim with stride 1 found, return 1
        return 1
    alignments = get_alignments(dtype)
    for alignment in alignments:
        if not a_factor_of(size[contiguous_dim], alignment) or not a_factor_of(
            offset, alignment
        ):
            continue
        if all(
            (dim == contiguous_dim)
            or a_factor_of(inductor_layout.stride[dim], alignment)
            for dim in range(len(size))
        ):
            return alignment
    return 1


# TODO (SYCL) : Add helpers for CUTLASS kernels testing & benchmarking once standalone
# runner is enabled.
