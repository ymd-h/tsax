"""
Informer (:mod:`tsax.informer`)
===============================

Notes
-----
This module implements Informer [1]_.

References
----------
.. [1] H. Zhou et al., "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting", AAAI 2021, Vol. 35, No. 12
   https://ojs.aaai.org/index.php/AAAI/article/view/17325,
   https://arxiv.org/abs/2012.07436
"""
from __future__ import annotations
from typing import Tuple
import math

import jax
import jax.numpy as jnp
from jax import Array
from jax.random import KeyArray
from jax.typing import ArrayLike
from flax import linen as nn

from .core import ResidualLayerNorm

__all__ = [
    "Informer",
    "EncoderStack",
    "DecoderStack",
    "EncoderLayer",
    "DecoderLayer",
    "Distlling",
    "MultiHeadAttention",
    "ProbSparseAttention",
    "FeedForward",
]


KERNEL_SIZE: int = 3
"""
Default Kernel Size for Conv1d at Distilling Layer
"""

EMBEDDING_ALPHA: float = 1.0
"""
Default Coefficient for Embedding Alpha.
This is recommended value if input sequence has been normalized.
"""

class Embedding(nn.Module):
    """
    Embedding Layer
    """
    dm: int
    Vs: Tuple[int, ...]
    kernel: int = KERNEL_SIZE
    alpha: float = EMBEDDING_ALPHA

    @nn.compact
    def __call__(self, seq: ArrayLike, cat: ArrayLike) -> Array:
        """
        Call Embedding Layer

        Parameters
        ----------
        seq : ArrayLike
            Numerical Sequence. [B, L, d_sequence]
        cat : ArrayLike
            Categorical Sequence for Temporal Information. [B, L, d_categorical]

        Returns
        -------
        embedded : Array
            Embedded. [B, L, dm]
        """
        assert seq.shape[:2] == cat.shape[:2], "BUG"
        assert len(Vs) == cat.shape[2], "BUG"

        L: int = seq.shape[1]

        conv = nn.Conv(features=self.dm, kernel_size=kernel)
        embedded = jnp.moveaxis(conv(jnp.moveaxis(seq, (1,), (-1,))), (-1,), (1,))
        assert embedded.shape == (*seq.shape[:2], self.dm)

        embedded = (embedded
                    .at[:].multiply(self.alpha)
                    .at[:].add(PositionalEncoding(dm=dm, L=L, Lfreq=2*L)(seq))
                    .at[:].add(CategoricalEncoding(Vs=self.Vs, dm=self.dm)(cat)))

        return embedded

class Distilling(nn.Module):
    """
    Distilling Layer
    """
    kernel: int = KERNEL_SIZE

    @nn.compact
    def __call__(self, x: ArrayLike) -> Array:
        """
        Call Distilling Layer

        Parameters
        ----------
        x : ArrayLike
            Inputs Sequence. [B, L, d]

        Returns
        -------
        x : Array
            Convoluted Sequence. [B, L/2, d]
        """
        x = nn.Conv(features=1, kernel_size=self.kernel)(x)
        x = nn.activation.elu(x)
        x = nn.max_pool(x, window_shape=(2,), strides=(2,))

        return x



class ProbSparseAttention(nn.Module):
    """
    ProbSparse self-attention
    """
    c: int

    @nn.compact
    def __call__(self,
                 Q: ArrayLike,
                 K: ArrayLike,
                 V: ArrayLike,
                 rng: KeyArray) -> Array:
        """
        Call ProbSparse self-attention

        Parameters
        ----------
        Q : ArrayLike
            Query. [B, L, d]
        K : ArrayLike
            Key. [B, L, d]
        V : ArrayLike
            Value. [B, L, d]
        rng : KeyArray
            Random Number Generator Key will be consumed.

        Returns
        -------
        A : Array
            ProbSparse self-attention
        """
        assert Q.shape[0] == K.shape[0] == V.shape[0], "BUG"
        assert Q.shape[2] == K.shape[2] == V.shape[2], "BUG"

        B: int = int(Q.shape[0])
        m: int = int(Q.shape[1])
        n: int = int(K.shape[1])
        d: int = int(Q.shape[2])

        u: int = int(self.c * math.log(m))
        U: int = int(     m * math.log(n))


        @jax.vmap
        def _each_sample(_Q, _K, _V, _rng):
            _Kbar = _K.at[jax.random.choice(_rng, U, replace=False),:].get()

            _Sbar = jnp.matmul(_Q, jnp.transpose(_Kbar, (1, 0)))
            _M = jnp.max(_Sbar, axis=1) - jnp.mean(_Sbar, axis=1)
            assert _M.shape == (m, U), "BUG"

            _, _I = jax.lax.top_k(_M, u)
            assert _I.shape == (u,), "BUG"

            _Qbar = _Q.at[_I, :].get()
            assert _Qbar.shape == (u, d), "BUG"

            _QbarK = jnp.matmul(_Qbar, jnp.transpose(_K, (1, 0)))
            assert _QbarK.shape == (u, n), "BUG"

            _S1 = jnp.matmul(nn.activation.softmax(_QbarK / math.sqrt(d)), _V)
            assert _S1.shape == (u, d), "BUG"

            _S = jnp.zeros((m, d), dtype=Q.dtype)
            _S = _S.at[:, :].set(jnp.mean(_V, axis=1, keepdims=True))
            _S = _S.at[_I, :].set(_S1)

            return _S

        rng_batch = jax.random.split(rng, B)

        S = _each_sample(Q, K, V, rng_batch)
        assert S.shape == (B, m, d), "BUG"

        return S


class MultiHeadAttention(nn.Module):
    """
    Multi Head Attention Layer

    Attributes
    ----------
    c : int
        Hyper Parameter
    nH : int
        Number of Multi Head
    dm : int
        Model Dimension
    """
    c: int
    nH: int = NH
    dm: int = DM
    Pdrop: float = PDROP

    @nn.compact
    def __call__(self,
                 Q: ArrayLike,
                 K: ArrayLike,
                 V: ArrayLike,
                 rng: KeyArray, *,
                 with_dropout: bool = False) -> Array:
        """
        Call Multi Head Attention

        Parameters
        ----------
        Q : ArrayLike
            Query. [B, L, dm]
        K : ArrayLike
            Key. [B, L, dm]
        V : ArrayLike
            Value. [B, L, dm]
        rng : KeyArray
            Random Number Generator Key will be consumed.
        with_dropout : bool, optional
            Whether dropout or not

        Returns
        -------
        MHA : Array
            Multi Head Attention. [B, L, dm]
        """
        assert K.shape == V.shape, "BUG"
        assert Q.shape[0] == K.shape[0], "BUG"
        assert Q.shape[2] == K.shape[2], "BUG"

        rngs = jax.random.split(rng, self.nH)

        # x: [B, L, dm (= dm/nH * nH)]
        d: int = self.dm // self.nH
        x = jnp.concatenate([ProbSparseAttention(name=f"head_{i}")(Q, K, V, rngs[i])
                             for i in range(self.nH)],
                            axis=2)
        assert x.shape == (*Q.shape[:2], d * self.nH), "BUG"

        MHA = nn.Dense(features=self.dm, use_bias=Fase, name="WO")(x)
        assert Q.shape == MHA.shape, "BUG"

        if with_dropout:
            MHA = MHA.at[:].set(nn.Dropout(self.Pdrop, deterministic=False)(MHA))

        return MHA



class Informer(nn.Module):
    """
    Informer

    Attributes
    ----------
    """
    I: int
    O: int
    Ltoken: int
    nE: int
    nD: int
    dff: int = DFF
    eps: float = EPS
    Pdrop: float = PDROP

    def setup(self):
        self.encoder = EncoderStack(N=self.nE,
                                    dm=self.dm,
                                    nH=self.nH,
                                    dff=self.dff,
                                    eps=self.eps,
                                    Pdrop=self.Pdrop)
        self.decoder = DecoderStack(N=self.N,
                                    dm=self.dm,
                                    nH=self.nH,
                                    dff=self.dff,
                                    eps=self.eps,
                                    Pdrop=self.Pdrop)

    def encode(self,
               inputs: ArrayLike, *,
               with_dropout: bool = False) -> Array:
        """
        Encode with Informer
        """
        assert inputs.shape[1:] == (self.I, self.dm), "BUG"

    def decode(self,
               inputs: ArrayLike,
               outputs: ArrayLike, *,
               with_dropout: bool = False) -> Array:
        """
        Decode with Informer
        """
        assert inputs.shape[0] == outputs.shape[0], "BUG"
        assert inputs.shape[1:] == (self.I, self.dm), "BUG"
        assert outputs.shape[1:] == (self.Ltoken + self.O, self.dm), "BUG"

    def __call__(self,
                 inputs: ArrayLike, *,
                 with_dropout: bool = False) -> Array:
        """
        Call Informer

        Parameters
        ----------
        inputs : ArrayLike
            Inputs Signal. [B, I, dm]
        with_dropout : bool, optional
            Whether dropout or not.

        Returns
        -------
        pred : Array
            Predicted Signal. [B, O, dm]
        """
        assert inputs.shape[1:] == (self.I, self.dm), "BUG"

        outputs = jnp.zeros((inputs.shape[0], self.Ltoken + self.O),
                            dtype=inputs.dtype)
        outputs.at[:,:inputs.shape[1],:].set(inputs)

        inputs = self.encode(inputs, with_dropout=with_dropout)

        pred = self.decode(inputs, outputs, with_dropout=with_dropout)

        return pred.at[:,pred.shape[1]-self.O:,:].get()
