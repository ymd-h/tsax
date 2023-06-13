"""
Tsax Typing (:mod:`tsax.typing`)
================================
"""
from __future__ import annotations
from typing import Dict, List, Tuple, TypeVar, Union

from jax import Array
from jax.typing import ArrayLike
from jax.random import KeyArray
from typing_extensions import Protocol

__all__ = [
    "Array",
    "ArrayLike",
    "KeyArray",
    "CarryT",
    "DataT",
    "SplitFn",
    "ActivationFn",
]


CarryT = TypeVar("CarryT")
DataT = TypeVar("DataT",
                bound=Union[Array, List[Array], Tuple[Array, ...]])


class SplitFn(Protocol):
    def __call__(self,
                 key: KeyArray, *,
                 train: bool=False) -> Tuple[KeyArray, Dict[str, KeyArray]]: ...


class ActivationFn(Protocol):
    def __call__(self, x: Array) -> Array: ...
