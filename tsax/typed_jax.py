"""
Typed Wapper for JAX/Flax (:mod:`tsax.typed_jax`)
=================================================
"""
from __future__ import annotations
from typing import cast, Callable, Iterable, Sequence, TypeVar, Union

import jax
from typing_extensions import ParamSpec


__all__ = [
    "jit",
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
