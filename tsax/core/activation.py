"""
Activation (:mod:`tsax.core.activation`)
========================================
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn

from tsax.typing import ArrayLike, Array


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
    return nn.activation.softmax(x)


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
    s = (jnp.sin(x + 0.25 * jnp.pi) ** 2)
    return s / jnp.sum(s, axis=-1, keepdims=True)


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
    return nn.activation.softmax(jnp.sin(x))
