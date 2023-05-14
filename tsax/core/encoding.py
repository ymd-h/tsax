from __future__ import annotations
from typing import Optional

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from flax import linen as nn


__all__ = [
    "PositionalEncoding",
]


def positional_encoding(dm: int,
                        L: int,
                        Lfreq: int, *,
                        dtype: Optional[jnp.dtype] = None) -> Array:
    """
    Create Positional Encoding

    Parameters
    ----------
    dm : int
        Model Dimension
    L : int
        Input Sequence Length
    Lfreq : int
        Frequency Normalization Length.
    dtype : jax.numpy.dtype, optional
        Data Type.

    Returns
    -------
    PE : Array
        Positional Encoding. [L, dm]
    """
    half: int = dm // 2
    sin_dim: int = half + (dm % 2)
    cos_dim: int = half

    freq = 1.0 / (Lfreq ** (2 * jnp.arange(sin_dim) / dm))
    theta = jax.vmap(lambda pos: pos * freq)(jnp.arange(L))

    PE = (jnp.zeros((L, dm), dtype=dtype)
          .at[:,0::2].set(jnp.sin(theta))
          .at[:,1::2].set(jnp.cos(theta.at[:,:cos_dim].get())))

    return PE


class PositionalEncoding:
    r"""
    Positional Encoding

    Notes
    -----
    ``PE(pos, 2j  ) = sin(pos / (Lfreq ** (2 * j / dm)))``
    ``PE(pos, 2j+1) = cos(pos / (Lfreq ** (2 * j / dm)))``
    """

    def __init__(self,
                 dm: int,
                 L: int, *,
                 Lfreq: int = 10000,
                 lazy: bool = False,
                 dtype: Optional[jnp.dtype] = None):
        """
        Initialize Positional Encoding

        Parameters
        ----------
        dm : int
            Model Dimension
        L : int
            Input Sequence Length
        Lfreq : int, optional
            Frequency Normalization Length. Default is ``10000``.
        lazy : bool, optional
            If ``False`` (default), prepare beforehand.
        dtype : jax.numpy.dtype, optional
            Data Type.
        """
        self.dm: int = dm
        self.L: int = L
        self.Lfreq = Lfreq
        self.lazy: bool = lazy

        if not self.lazy:
            self.pe = positional_encoding(dm=self.dm,
                                          L=self.L,
                                          Lfreq=self.Lfreq,
                                          dtype=dtype)

    def __call__(self, x: ArrayLike) -> Array:
        """
        Positional Encoding

        Parameters
        ----------
        x : ArrayLike
            Inputs. [B, L, dm]

        Returns
        -------
        PE : Array
            Positional Encoding. [L, dm]
        """
        assert x.shape[1:] == (self.L, self.dm), "BUG"

        if self.lazy:
            return positional_encoding(dm=self.dm,
                                       L=self.L,
                                       Lfreq=self.Lfreq,
                                       dtype=x.dtype)

        return self.pe
