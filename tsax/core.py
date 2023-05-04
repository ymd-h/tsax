"""
Core (mod:`tsax.core`)
======================
"""
from __future__ import annotations
from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
import flax.linen as nn


class ResidualLayerNorm(nn.Module):
    """
    Residual Layer Normalization

    Attributes
    ----------
    sublayer : callable
        Sub Layer
    eps : float
        Small Positive Value for Layer Normalization
    """
    sublayer: Callable[[ArrayLike], Array]
    eps: float

    @nn.compact
    def __call__(x: ArrayLike) -> Array:
        """
        Residual Connection followed by Layer Normalization

        Parameters
        ----------
        x : ArrayLike
            Input for Residual Connection and Sub Layer

        Returns
        -------
        x : Array
            Layer Normalized
        """
        return nn.LayerNorm(epsilon=self.eps)(x + self.sublayer(x))
