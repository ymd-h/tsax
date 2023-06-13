from __future__ import annotations
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

from tsax.core import ConvSeq
from tsax.typing import Array, ArrayLike


__all__ = [
    "PositionalEncoding",
    "CategoricalEncoding",
    "Embedding",
    "EMBEDDING_ALPHA",
]

EMBEDDING_ALPHA: float = 1.0
"""
Default Coefficient for Embedding Alpha.
This is recommended value if input sequence has been normalized.
"""


def positional_encoding(dm: int,
                        L: int,
                        Lfreq: int, *,
                        shift: int = 0,
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
    shift : int
        Position Offset Shift.
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
    theta = jax.vmap(lambda pos: (pos + shift) * freq)(jnp.arange(L))

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

    def __call__(self, x: ArrayLike, shift: int = 0) -> Array:
        """
        Positional Encoding

        Parameters
        ----------
        x : ArrayLike
            Inputs. [B, L, dm]
        shift : int
            Position Offset Shift.
            Used only if ``lazy`` is ``True``.

        Returns
        -------
        PE : Array
            Positional Encoding. [L, dm]
        """
        x = jnp.asarray(x)
        assert x.shape[1:] == (self.L, self.dm), f"BUG: {x.shape}"

        if self.lazy:
            return positional_encoding(dm=self.dm,
                                       L=self.L,
                                       Lfreq=self.Lfreq,
                                       shift=shift,
                                       dtype=x.dtype)

        return self.pe


class CategoricalEncoding(nn.Module):
    """
    Categorical Encoding

    Attributes
    ----------
    Vs : tuple of ints
        Vocabulary Sizes
    dm : int
        Model Dimension

    Notes
    -----
    This encoding is designed for Temporal Embedding at Informer [1]_.

    References
    ----------
    .. [1] H. Zhou et al., "Informer: Beyond Efficient Transformer
       for Long Sequence Time-Series Forecasting", AAAI 2021, Vol. 35, No. 12
       https://ojs.aaai.org/index.php/AAAI/article/view/17325,
       https://arxiv.org/abs/2012.07436

    Examples
    --------
    >>> from tsax.core import CategoricalEncoding

    If ``x`` has "month" (``0`` to ``11``) and "date" (``0`` to ``30``)
    and model dimension is ``5``, then encoding becomes;

    >>> enc = CategoricalEncoding(Vs=(12, 31), dm=5)
    """
    Vs: Tuple[int, ...]
    dm: int

    @nn.compact
    def __call__(self, x: ArrayLike) -> Array:
        """
        Call Categorical Encoding

        Parameters
        ----------
        x : ArrayLike
            Input Categorical Sequence. [B, L, C]

        Returns
        -------
        embedded : Array
            Embedded. [B, L, dm]
        """
        x = jnp.asarray(x)
        assert x.shape[-1] == len(self.Vs), "BUG"

        embedded = jnp.sum(
            jnp.stack([nn.Embed(v, self.dm)(x.at[:,:,i].get())
                       for i, v in enumerate(self.Vs)]),
            axis=0
        )

        assert embedded.shape == (*x.shape[:-1], self.dm), "BUG"

        return embedded


class Embedding(nn.Module):
    """
    Embedding Layer

    Attributes
    ----------
    dm : int
        Model Dimension
    Vs : tuple of ints
        Vocabulary Size for each Categorical Dimension
    kernel : int
        Kernel Size for 1d Convolution
    alpha : float
        Coefficient for Input Sequence
    Pdrop : float
        Dropout Probability
    with_positional : bool
        Whether use positional encoding or not
    """
    dm: int
    Vs: Tuple[int, ...]
    kernel: int
    alpha: float
    Pdrop: float
    with_positional: bool

    @nn.compact
    def __call__(self,
                 seq: ArrayLike,
                 cat: Optional[ArrayLike] = None, *,
                 with_dropout: bool = False) -> Array:
        """
        Call Embedding Layer

        Parameters
        ----------
        seq : ArrayLike
            Numerical Sequence. [B, L, d_seq]
        cat : ArrayLike, optional
            Categorical Sequence for Temporal Information. [B, L, d_cat]
        with_dropout : bool
            Whether dropout or not

        Returns
        -------
        embedded : Array
            Embedded. [B, L, dm]
        """
        seq = jnp.asarray(seq)
        cat = jnp.asarray(cat) if cat is not None else None
        fmt = f"BUG: {seq.shape} vs {cat.shape if cat is not None else None}"
        assert (cat is None) or (seq.shape[0] == cat.shape[0]), fmt
        assert (cat is None) or (seq.shape[1] >= cat.shape[1]), fmt
        assert (cat is None) or (len(self.Vs) == cat.shape[2]), fmt

        L: int = seq.shape[1]

        # Token Embedding as Projector
        embedded = ConvSeq(dm=self.dm, kernel=self.kernel)(seq)
        assert embedded.shape == (*seq.shape[:2], self.dm), f"BUG: {embedded.shape}"

        embedded = embedded.at[:].multiply(self.alpha)

        # Positional Encoding
        if self.with_positional:
            embedded = (
                embedded
                .at[:].add(PositionalEncoding(dm=self.dm, L=L, Lfreq=2*L)(embedded))
            )

        # Categorical Encoding designed for Temporal Embedding
        if cat is not None:
            S: int = cat.shape[1]
            embedded = (
                embedded
                .at[:,:S,:].add(CategoricalEncoding(Vs=self.Vs, dm=self.dm)(cat))
            )

        if with_dropout:
            embedded = embedded.at[:].set(
                nn.Dropout(self.Pdrop, deterministic=False)(embedded)
            )

        return embedded
