from __future__ import annotations
from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import Array
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
    sublayer: Callable[[Array], Array]
    eps: float

    @nn.compact
    def __call__(self, x: Array) -> Array:
        """
        Residual Connection followed by Layer Normalization

        Parameters
        ----------
        x : Array
            Input for Residual Connection and Sub Layer

        Returns
        -------
        x : Array
            Layer Normalized
        """
        return cast(Array, nn.LayerNorm(epsilon=self.eps)(x + self.sublayer(x)))
