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
from typing import Optional, Tuple
import math

import jax
import jax.numpy as jnp
from jax import Array
from jax.random import KeyArray
from jax.typing import ArrayLike
from flax import linen as nn

from tsax.core import (
    ConvSeq,
    PositionalEncoding,
    CategoricalEncoding,
    ResidualLayerNorm,
    SubsequentMask,
)

__all__ = [
    "Informer",
    "EncoderStack",
    "DecoderStack",
    "EncoderLayer",
    "DecoderLayer",
    "Distilling",
    "MultiHeadAttention",
    "ProbSparseAttention",
    "Embedding",
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

DFF: int = 2048
"""
Default Number of Units at Hidden Layer of Feed Forward
"""

NH: int = 8
"""
Default Number of Multi Head Attention
"""

EPS: float = 1e-12
"""
Default Value for Small Positive Value for Layer Normalization
"""

PDROP: float = 0.1
"""
Default Probability of Dropout Rate
"""

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
    """
    dm: int
    Vs: Tuple[int, ...] = tuple()
    kernel: int = KERNEL_SIZE
    alpha: float = EMBEDDING_ALPHA

    @nn.compact
    def __call__(self, seq: ArrayLike, cat: Optional[ArrayLike] = None) -> Array:
        """
        Call Embedding Layer

        Parameters
        ----------
        seq : ArrayLike
            Numerical Sequence. [B, L, d_sequence]
        cat : ArrayLike, optional
            Categorical Sequence for Temporal Information. [B, L, d_categorical]

        Returns
        -------
        embedded : Array
            Embedded. [B, L, dm]
        """
        assert (cat is None) or (seq.shape[:2] == cat.shape[:2]), "BUG"
        assert (cat is None) or (len(self.Vs) == cat.shape[2]), "BUG"

        L: int = seq.shape[1]

        embedded = ConvSeq(dm=self.dm, kernel=self.kernel)(seq)
        assert embedded.shape == (*seq.shape[:2], self.dm), f"BUG: {embedded.shape}"

        if cat is not None:
            embedded = (embedded
                        .at[:].multiply(self.alpha)
                        .at[:].add(PositionalEncoding(dm=self.dm,
                                                      L=L,
                                                      Lfreq=2*L)(embedded))
                        .at[:].add(CategoricalEncoding(Vs=self.Vs,
                                                       dm=self.dm)(cat)))

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
        B, L, d = x.shape
        x = ConvSeq(dm=d, kernel=self.kernel)(x)
        assert x.shape == (B, L, d), "BUG"

        x = nn.activation.elu(x)

        is_odd: int = L % 2
        x_pad = jnp.pad(x, ((0, 0), (1, is_odd), (0, 0)), constant_values=-jnp.inf)
        x = jax.vmap(
            lambda i: jnp.max(jax.lax.dynamic_slice(x_pad, (0, i, 0), (B, 3, d)),
                              axis=1),
            out_axes=1
        )(jnp.arange(0, L + is_odd, 2))
        assert x.shape == (B, (L + 1) // 2, d), "BUG"

        return x



class ProbSparseAttention(nn.Module):
    """
    ProbSparse self-attention

    Attributes
    ----------
    c : int
        Hyper-Parameter for Sampling Factor
    """
    c: int
    dk: int
    dv: int
    mask: bool = False
    rng_collection: str = "attention"

    @nn.compact
    def __call__(self,
                 Q: ArrayLike,
                 K: ArrayLike,
                 V: ArrayLike) -> Array:
        """
        Call ProbSparse self-attention

        Parameters
        ----------
        Q : ArrayLike
            Query. [B, LQ, dm]
        K : ArrayLike
            Key. [B, LK, dm]
        V : ArrayLike
            Value. [B, LK, dm]

        Returns
        -------
        A : Array
            ProbSparse self-attention. [B, L, dv]
        """
        assert Q.shape[0] == K.shape[0] == V.shape[0], "BUG"
        assert K.shape[1] == V.shape[1], "BUG"
        assert Q.shape[2] == K.shape[2] == V.shape[2], "BUG"

        B: int = int(Q.shape[0])
        m: int = int(Q.shape[1])
        n: int = int(K.shape[1])
        d: int = int(Q.shape[2])

        # Note: Official implementation is different from the paper.
        #       We obey the implementation.
        u: int = min(int(self.c * math.ceil(math.log(m))), m)
        U: int = min(int(self.c * math.ceil(math.log(n))), n)

        Q = nn.Dense(features=self.dk, name="WQ")(Q)
        K = nn.Dense(features=self.dk, name="WK")(K)
        V = nn.Dense(features=self.dv, name="WV")(V)

        if self.mask:
            mask = SubsequentMask(n)

        @jax.vmap
        def _each_sample(_Q, _K, _V, _rng):
            _Kbar = _K.at[jax.random.choice(_rng,
                                            _K.shape[0],
                                            shape=(U,),
                                            replace=False),:].get()

            _Sbar = jnp.matmul(_Q, jnp.transpose(_Kbar, (1, 0)))
            _M = jnp.max(_Sbar, axis=1) - jnp.mean(_Sbar, axis=1)
            assert _M.shape == (m,), f"BUG: {_M.shape} != {(m,)}"

            _, _I = jax.lax.top_k(_M, u)
            assert _I.shape == (u,), "BUG"

            _Qbar = _Q.at[_I, :].get()
            assert _Qbar.shape == (u, self.dk), "BUG"

            _QbarK = jnp.matmul(_Qbar, jnp.transpose(_K, (1, 0)))
            assert _QbarK.shape == (u, n), "BUG"

            if self.mask:
                _QbarK = _QbarK.at[:].set(
                    jnp.where(mask.at[_I,:].get()==1, _QbarK, -jnp.inf)
                )

            _S1 = jnp.matmul(nn.activation.softmax(_QbarK / math.sqrt(d)), _V)
            assert _S1.shape == (u, self.dv), "BUG"

            _S = jnp.zeros((m, self.dv), dtype=Q.dtype)
            _S = _S.at[:, :].set(
                jnp.cumsum(_V, axis=1) if self.mask
                else jnp.mean(_V, axis=1, keepdims=True)
            )
            _S = _S.at[_I, :].set(_S1)

            return _S

        rng = jax.random.split(self.make_rng(self.rng_collection), B)

        S = _each_sample(Q, K, V, rng)
        assert S.shape == (B, m, self.dv), "BUG"

        return S


class MultiHeadAttention(nn.Module):
    """
    Multi Head Attention Layer

    Attributes
    ----------
    c : int
        Hyper Parameter of Sampling Factor
    nH : int
        Number of Multi Head
    dm : int
        Model Dimension
    """
    c: int
    dm: int
    nH: int = NH
    Pdrop: float = PDROP
    mask: bool = False

    @nn.compact
    def __call__(self,
                 Q: ArrayLike,
                 K: ArrayLike,
                 V: ArrayLike, *,
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

        # x: [B, L, dm (= dm/nH * nH)]
        d: int = self.dm // self.nH
        x = jnp.concatenate([ProbSparseAttention(c=self.c,
                                                 mask=self.mask,
                                                 name=f"head_{i}")(Q, K, V)
                             for i in range(self.nH)],
                            axis=2)
        assert x.shape == (*Q.shape[:2], d * self.nH), "BUG"

        MHA = nn.Dense(features=self.dm, use_bias=False, name="WO")(x)
        assert Q.shape == MHA.shape, "BUG"

        if with_dropout:
            MHA = MHA.at[:].set(nn.Dropout(self.Pdrop, deterministic=False)(MHA))

        return MHA


class EncoderLayer(nn.Module):
    pass

class DecoderLayer(nn.Module):
    pass

class EncoderStack(nn.Module):
    pass

class DecoderStack(nn.Module):
    pass

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
