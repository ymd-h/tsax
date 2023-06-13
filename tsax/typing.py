"""
Tsax Typing (:mod:`tsax.typing`)
================================
"""
from __future__ import annotations
from typing import List, Tuple, TypeVar, Union

from jax import Array
from jax.typing import ArrayLike
from jax.random import KeyArray

CarryT = TypeVar("CarryT")
DataT = TypeVar("DataT",
                bound=Union[ArrayLike, List[ArrayLike], Tuple[ArrayLike, ...]])

__all__ = [
    "Array",
    "ArrayLike",
    "KeyArray",
    "CarryT",
    "DataT",
]
