"""
Typed Wapper for JAX/Flax (:mod:`tsax.typed_jax`)
=================================================
"""
from __future__ import annotations
import functools
from typing import (
    cast,
    Any,
    Callable,
    Hashable,
    Iterable,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import jax
import flax.linen as nn
from typing_extensions import ParamSpec

from tsax.typing import Array


__all__ = [
    "jit",
    "value_and_grad",
    "vmap",
    "relu",
    "gelu",
    "Dense",
]


P = ParamSpec("P")
T = TypeVar("T")


@functools.wraps(jax.jit)
def jit(f: Callable[P, T], *,
        static_argnums: Union[None, int, Sequence[int]]=None,
        static_argnames: Union[str, Iterable[str], None]=None) -> Callable[P, T]:
    return cast(
        Callable[P, T],
        jax.jit(f, static_argnums=static_argnums, static_argnames=static_argnames)
    )


@functools.wraps(jax.value_and_grad)
def value_and_grad(f: Callable[P, T], *,
                   argnums: Union[int, Sequence[int]]=0,
                   has_aux: bool=False,
                   holomorphic: bool=False,
                   allow_int: bool=False,
                   reduce_axes: Sequence[Hashable]=()) -> Callable[P, Tuple[T, T]]:
    return cast(
        Callable[P, Tuple[T, T]],
        jax.value_and_grad(f,
                           argnums=argnums, has_aux=has_aux,
                           holomorphic=holomorphic, reduce_axes=reduce_axes)
    )


@functools.wraps(jax.vmap)
def vmap(
        f: Callable[P, T], *,
        in_axes: Union[int, None, Sequence[Any]]=0,
        out_axes: Any=0,
        axis_name: Optional[Hashable]=None,
        axis_size: Optional[int]=None,
        spmd_axis_name: Optional[Union[Hashable, Tuple[Hashable, ...]]]=None
) -> Callable[P, T]:
    return cast(Callable[P, T],
                jax.vmap(f,
                         in_axes=in_axes, out_axes=out_axes,
                         axis_name=axis_name, axis_size=axis_size,
                         spmd_axis_name=spmd_axis_name))

@functools.wraps(nn.activation.relu)
def relu(x: Array) -> Array:
    return cast(Array, nn.activation.relu(x))

@functools.wraps(nn.activation.gelu)
def gelu(x: Array, approximate: bool=True) -> Array:
    return cast(Array, nn.activation.gelu(x, approximate=approximate))


@functools.wraps(nn.Dense)
def Dense(features: int,
          use_bias: bool=True,
          name: Optional[str]=None) -> Callable[[Array], Array]:
    return nn.Dense(features=features, use_bias=use_bias, name=name) # type: ignore[call-arg]
