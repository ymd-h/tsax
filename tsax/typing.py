"""
Tsax Typing (:mod:`tsax.typing`)
================================
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

from jax import Array
from jax.typing import ArrayLike
from jax.random import KeyArray
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


CarryT = TypeVar("CarryT")
DataT = TypeVar("DataT",
                bound=Union[Array, List[Array], Tuple[Array, ...]])

ModelParam: TypeAlias = fcore.FrozenDict[str, Any]

LayerNormMode: TypeAlias = Literal["post", "pre"]

class ModelCall(Protocol):
    def __call__(self,
                 variables: ModelParam,
                 data: DataT,
                 rngs: Union[KeyArray, Dict[str, KeyArray], None]=None,
                 train: bool = False) -> Array: ...


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
