"""
Tsax Typing (:mod:`tsax.typing`)
================================
"""
from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional, Tuple, TypeVar, Union

import jax
from jax import Array as jaxArray
import jax.numpy as jnp
from jax.typing import ArrayLike as jaxArrayLike
from jax.random import KeyArray as jaxKeyArray
from jax._src.typing import Shape
from jax._src.numpy.lax_numpy import _ScalarMeta
from flax import core as fcore
from typing_extensions import Never, Protocol, TypeAlias

__all__ = [
    "Array",
    "ArrayLike",
    "KeyArray",
    "CarryT",
    "DataT",
    "LayerNormMode",
    "ModelParam",
    "ModelCall",
    "SplitFn",
    "ActivationFn",
    "SoftmaxLike",
    "CallNever",
]

# For Sphinx AutoSummary, we re-define JAX's typing

Array: TypeAlias = jaxArray
Array.__doc__ = """Array"""

ArrayLike: TypeAlias = jaxArrayLike
ArrayLike.__doc__ = """Array Like"""

KeyArray: TypeAlias = jaxKeyArray
KeyArray.__doc__ = """Key Array"""

CarryT = TypeVar("CarryT")
DataT = TypeVar("DataT",
                bound=Union[Array, List[Array], Tuple[Array, ...]])

ModelParam: TypeAlias = Union[Dict[str, fcore.FrozenDict[str, Any]],
                              fcore.FrozenDict[str, Any]]
"""Model Parameter"""

LayerNormMode: TypeAlias = Literal["post", "pre"]
"""Layer Normalization Mode"""


Dtype: TypeAlias = Union[
    jnp.dtype,
    str,
    _ScalarMeta
]
Dtype.__doc__ = """Data Type"""


PrecisionLike: TypeAlias = Union[None,
                                 str, jax.lax.Precision,
                                 Tuple[str, str],
                                 Tuple[jax.lax.Precision, jax.lax.Precision]]
PrecisionLike.__doc__ = """Presicion Like"""

class ModelCall(Protocol):
    def __call__(self,
                 variables: ModelParam,
                 data: DataT,
                 rngs: Union[KeyArray, Dict[str, KeyArray], None]=None,
                 train: bool = False,
                 mutable: Optional[List[str]] = None) -> Array: ...


class SplitFn(Protocol):
    def __call__(self,
                 key: KeyArray, *,
                 train: bool=False) -> Tuple[KeyArray, Dict[str, KeyArray]]: ...


class ActivationFn(Protocol):
    def __call__(self, x: Array) -> Array: ...


class SoftmaxLike(Protocol):
    def __call__(self, x: ArrayLike) -> Array: ...


def CallNever(_: Never) -> Never:
    raise AssertionError("BUG")
