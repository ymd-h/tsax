"""
Tsax Typing (:mod:`tsax.typing`)
================================
"""
from __future__ import annotations
from typing import TypeVar

from jax import Array
from jax.typing import ArrayLike
from jax.random import KeyArray

CarryT = TypeVar("Carry")

__all__ = [
    "Array",
    "ArrayLike",
    "KeyArray",
    "CarryT",
]
