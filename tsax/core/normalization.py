from __future__ import annotations
from typing import cast, Callable, Literal

import jax
import jax.numpy as jnp
import flax.linen as nn

from tsax.typing import Array, CallNever, LayerNormMode

__all__ = [
    "ResidualLayerNorm",
]


class ResidualLayerNorm(nn.Module):
    """
    Residual Layer Normalization

    Parameters
    ----------
    sublayer : callable
        Sub Layer
    eps : float
        Small Positive Value for Layer Normalization
    """
    sublayer: Callable[[Array], Array]
    eps: float
    position: LayerNormMode = "post"

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
        LN = nn.LayerNorm(epsilon=self.eps)
        if self.position == "post":
            return cast(Array, LN(x + self.sublayer(x)))
        elif self.position == "pre":
            return x + self.sublayer(cast(Array, LN(x)))
        else:
            CallNever(self.position)
