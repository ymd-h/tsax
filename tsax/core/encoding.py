from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from flax import linen as nn


__all__ = [
    "PositionalEncoding",
]


class PositionalEncoding(nn.Module):
    r"""
    Positional Encoding

    Attributes
    ----------
    dm : int
        Model Dimension
    L : int
        Length

    Notes
    -----
    ``PE(pos, 2j  ) = sin(pos / (L ** (2 * j / dm)))``
    ``PE(pos, 2j+1) = cos(pos / (L ** (2 * j / dm)))``
    """
    dm: int
    L: int

    def setup(self) -> None:
        half: int = self.dm // 2
        self.sin_dim: int = half + (self.dm % 2)
        self.cos_dim: int = half

        self.freq = 1.0 / (self.L ** (2 * jnp.arange(self.sin_dim) / self.dm))

    def __call__(self, x: ArrayLike) -> Array:
        """
        Positional Encoding

        Parameters
        ----------
        x : ArrayLike
            Inputs. [B, Lx, dm]

        Returns
        -------
        PE : Array
            Positional Encoding. [Lx, dm]
        """
        assert x.shape[2] == self.dm, "BUG"

        Lx: int = x.shape[1]
        theta = jax.vmap(lambda pos: pos * self.freq)(jnp.arange(Lx))

        PE = jnp.zeros((Lx, self.dim), dtype=x.dtype)

        PE = (PE
              .at[:,0::2].set(jnp.sin(theta))
              .at[:,1::2].set(jnp.cos(theta.at[:,:self.cos_dim].get())))

        return PE
