"""
Typed Wapper for JAX/Flax (:mod:`tsax.typed_jax`)
=================================================
"""
from __future__ import annotations
from typing import cast, Callable, Iterable, Sequence, TypeVar, Union

import jax
import flax.linen as nn
from typing_extensions import ParamSpec

from tsax.typing import Array


__all__ = [
    "jit",
    "relu",
    "gelu",
]


P = ParamSpec("P")
T = TypeVar("T")


def jit(f: Callable[P, T], *,
        static_argnums: Union[None, int, Sequence[int]]=None,
        static_argnames: Union[str, Iterable[str], None]=None) -> Callable[P, T]:
    return cast(
        Callable[P, T],
        jax.jit(f, static_argnums=static_argnums, static_argnames=static_argnames)
    )

def relu(x: Array) -> Array:
    return cast(Array, nn.activation.relu(x))

def gelu(x: Array, approximate: bool=True) -> Array:
    return cast(Array, nn.activation.gelu(x, approximate=approximate))
