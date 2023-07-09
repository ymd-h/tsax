"""
Activation (:mod:`tsax.core.activation`)
========================================
"""
from __future__ import annotations
from typing import cast

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn

from tsax.typing import ArrayLike, Array

__all__ = [
    "Softmax",
    "Sin2MaxShifted",
    "SinSoftmax",
]

def Softmax(x: ArrayLike) -> Array:
    """
    Softmax

    Parameters
    ----------
    x : ArrayLike
        Value

    Returns
    -------
    x : Array
        Activated Value
    """
    return cast(Array, nn.activation.softmax(x))


def Sin2MaxShifted(x: ArrayLike) -> Array:
    """
    Sin2-max-shifted

    Parameters
    ----------
    x : ArrayLike
        Value

    Returns
    -------
    x : Array
        Activated Value
    """
    s = (jnp.where(x != -jnp.inf, jnp.sin(x + 0.25 * jnp.pi), 0) ** 2)
    return s / (1e-8 + jnp.sum(s, axis=-1, keepdims=True))


def SinSoftmax(x: ArrayLike) -> Array:
    """
    Sin-Softmax

    Parameters
    ----------
    x : ArrayLike
        Value

    Returns
    -------
    x : Array
        Activated Value
    """
    return cast(Array,
                nn.activation.softmax(jnp.where(x != -jnp.inf, jnp.sin(x), -jnp.inf)))
