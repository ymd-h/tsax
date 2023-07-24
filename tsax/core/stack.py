"""
Layer Stack (:mod:`tsax.core.stack`)
====================================
"""
from __future__ import annotations
from typing import cast, Callable, Optional, TypeVar, Tuple

import jax
import flax.linen as nn
from typing_extensions import TypeAlias

from tsax.typing import Array, DataT


__all__ = [
    "LayerStack"
]

T = TypeVar("T")
Layer = TypeVar("Layer", bound=nn.Module)

def LayerStack(layer: Layer,
               x: DataT,
               n: int,
               f: Optional[Callable[[Layer, DataT], DataT]]=None) -> DataT:
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

    Returns
    -------
    x : DataT
        Data processed Stack of Layers

    Notes
    -----
    This function is a convenient wrapper of ``flax.linen.scan``.
    """
    if f is None:
       def F(L: Layer, carry: DataT, _: None) -> Tuple[DataT, None]:
           return L(carry), None
    else:
        def F(L: Layer, carry: DataT, _: None) -> Tuple[DataT, None]:
            return cast(Callable[[Layer, DataT], DataT], f)(L, carry), None

    x, _ = nn.scan(
        F,
        variable_axes={"params": 0, "sigma_reparam": 0},
        variable_broadcast=False,
        variable_carry=False,
        split_rngs={"params": True,
                    "dropout": True,
                    "attention": True},
        length=n
    )(layer, x, None)

    return x
