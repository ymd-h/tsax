"""
Layer Stack (:mod:`tsax.core.stack`)
====================================
"""
from __future__ import annotations
from typing import Callable, TypeVar, Tuple

import jax
import flax.linen as nn
from typing_extensions import TypeAlias

from tsax.typing import Array, DataT


__all__ = [
    "LayerStack"
]

C = TypeVar("C", bound=DataT)
T = TypeVar("T")


def LayerStack(layer: nn.Module,
               x: C,
               n: int,
               f: Optional[Callable[[nn.Module, C], C]]=None) -> Array:
    """
    Call Identical Layer Stack

    Parameters
    ----------
    layer : nn.Module
        Encoder / Decoder Layer
    x : DataT
        Carried Data
    n : int
        Number of Layers
    f : Callable[[nn.Module, DataT], DataT], optional
        Conversion function when I/O signature of ``layer``
        is different from ``x``
    """
    if f is None:
       def F(L: nn.Module, carry: C, _: None) -> Tuple[C, None]:
           return L(carry), None
    else:
        def F(L: nn.Module, carry: C, _: None) -> Tuple[C, None]:
            return f(L, carry), None

    x, _ = nn.scan(
        F,
        variable_axes={"params": 0},
        variable_broadcast=False,
        variable_carry=False,
        split_rngs={"params": True,
                    "dropout": True,
                    "attention": True},
        length=self.nD
    )(layer, x, None)

    return x
