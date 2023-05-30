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
import wblog

from tsax.core import (
    Model,
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
    "MultiHeadProbSparseAttention",
    "Attention",
    "ProbSparseAttention",
    "Embedding",
    "FeedForward",
]


logger = wblog.getLogger()


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

NE: int = 3
"""
Default Number of Encoder Layers
"""

ND: int = 2
"""
Default Number of Decoder Layers
"""


EPS: float = 1e-12
"""
Default Value for Small Positive Value for Layer Normalization
"""

PDROP: float = 0.1
"""
Default Probability of Dropout Rate
"""


class FeedForward(nn.Module):
    """
    Feed Forward Layer

    Attributes
    ----------
    dff : int
        Hidden Layer Units
    Pdrop : float
        Probability of Dropout
    """
    dff: int = DFF
    Pdrop: float = PDROP

    @nn.compact
    def __call__(self, x: ArrayLike, *, with_dropout = False) -> Array:
        """
        Call Feed Foward Network

        Parameters
        ----------
        x : ArrayLike
            Inputs. [B, L, dm]
        with_dropout : bool, optional
            Whether dropout or not

        Returns
        -------
        y : Array
            Outputs. [B, L, dm]
        """
        B, L, dm = x.shape

        # h: [B, L, dff]
        h = nn.activation.relu(ConvSeq(dm=self.dff, kernel=1)(x))
        assert h.shape == (B, L, self.dff), "BUG"

        if with_dropout:
            h = h.at[:].set(nn.Dropout(self.Pdrop, deterministic=False)(h))

        y = ConvSeq(dm=dm, kernel=1)(h)
        assert y.shape == (B, L, dm), "BUG"

        if with_dropout:
            y = y.at[:].set(nn.Dropout(self.Pdrop, deterministic=False)(y))

        return y

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
    Pdrop: float = PDROP

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
            Numerical Sequence. [B, L, d_sequence]
        cat : ArrayLike, optional
            Categorical Sequence for Temporal Information. [B, L, d_categorical]
        with_dropout : bool
            Whether dropout or not

        Returns
        -------
        embedded : Array
            Embedded. [B, L, dm]
        """
        assert (cat is None) or (seq.shape[:2] == cat.shape[:2]), "BUG"
        assert (cat is None) or (len(self.Vs) == cat.shape[2]), "BUG"

        L: int = seq.shape[1]

        # Token Embedding as Projector
        embedded = ConvSeq(dm=self.dm, kernel=self.kernel)(seq)
        assert embedded.shape == (*seq.shape[:2], self.dm), f"BUG: {embedded.shape}"

        # Positional Embedding
        embedded = (
            embedded
            .at[:].multiply(self.alpha)
            .at[:].add(PositionalEncoding(dm=self.dm, L=L, Lfreq=2*L)(embedded))
        )

        if cat is not None:
            # Temporal Embedding
            embedded = (
                embedded.at[:].add(CategoricalEncoding(Vs=self.Vs, dm=self.dm)(cat))
            )

        if with_dropout:
            embedded = embedded.at[:].set(
                nn.Dropout(self.Pdrop, deterministic=False)(embedded)
            )

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


class Attention(nn.Module):
    """
    Attention Layer

    Attributes
    ----------
    dk : int
        Key Dimension
    dv : int
        Value Dimension
    """
    dk: int
    dv: int

    @nn.compact
    def __call__(self,
                 Q: ArrayLike,
                 K: ArrayLike,
                 V: ArrayLike) -> Array:
        """
        Call Attention Layer

        Parameters
        ----------
        Q : ArrayLike
            Query. [B, Lq, dm]
        K : ArrayLike
            Key. [B, Lk, dm]
        V : ArrayLike
            Value. [B, Lk, dm]

        Returns
        -------
        A : Array
            Attention. [B, Lk, dv]
        """
        assert K.shape == V.shape, "BUG"
        assert Q.shape[0] == K.shape[0], "BUG"
        assert Q.shape[2] == K.shape[2], "BUG"

        # Q, K: [B, L, dm] -> [B, L, dk]
        Q = nn.Dense(features=self.dk, name="WQ")(Q)
        K = nn.Dense(features=self.dk, name="WK")(K)

        # V: [B, L, dm] -> [B, L, dv]
        V = nn.Dense(features=self.dv, name="WV")(V)

        # QK^T: [B, Lk, Lq]
        QK: Array = jnp.matmul(Q, jnp.transpose(K, (0, 2, 1)))
        assert QK.shape == (*Q.shape[:2], K.shape[1]), "BUG"

        QK = QK.at[:].divide(jnp.sqrt(self.dk))

        # A: [B, Lk, dv]
        A: Array = jnp.matmul(nn.activation.softmax(QK), V)
        assert A.shape == (*Q.shape[:2], self.dv), "BUG"

        return A


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
    mask: int = 0
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
            mask = SubsequentMask(n).at[:,self.mask:].set(0)

        @jax.vmap
        def _each_sample(_Q, _K, _V, _rng):
            _Kbar = _K.at[jax.random.choice(_rng,
                                            _K.shape[0],
                                            shape=(U,),
                                            replace=False),:].get()
            assert _Kbar.shape == (U, self.dk), "BUG"

            _Sbar = jnp.matmul(_Q, jnp.transpose(_Kbar, (1, 0)))
            assert _Sbar.shape == (m, U), "BUG"

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
    nH : int
        Number of Multi Head
    dm : int
        Model Dimension
    """
    dm: int
    nH: int = NH
    Pdrop: float = PDROP

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
        x = jnp.concatenate([Attention(dk=d,
                                       dv=d,
                                       name=f"head_{i}")(Q, K, V)
                             for i in range(self.nH)],
                            axis=2)
        assert x.shape == (*Q.shape[:2], d * self.nH), "BUG"

        MHA = nn.Dense(features=self.dm, name="WO")(x)
        assert Q.shape == MHA.shape, "BUG"

        if with_dropout:
            MHA = MHA.at[:].set(nn.Dropout(self.Pdrop, deterministic=False)(MHA))

        return MHA


class MultiHeadProbSparseAttention(nn.Module):
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
    mask: int = 0

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
                                                 dk=d,
                                                 dv=d,
                                                 mask=self.mask,
                                                 name=f"head_{i}")(Q, K, V)
                             for i in range(self.nH)],
                            axis=2)
        assert x.shape == (*Q.shape[:2], d * self.nH), "BUG"

        MHA = nn.Dense(features=self.dm, name="WO")(x)
        assert Q.shape == MHA.shape, "BUG"

        if with_dropout:
            MHA = MHA.at[:].set(nn.Dropout(self.Pdrop, deterministic=False)(MHA))

        return MHA


class EncoderLayer(nn.Module):
    """
    Encoder Layer
    """
    c: int
    dm: int
    nH: int = NH
    dff: int = DFF
    eps: float = EPS
    Pdrop: float = PDROP

    @nn.compact
    def __call__(self,
                 inputs: ArrayLike, *,
                 with_dropout: bool = False) -> Array:
        """
        Call Encoder Layer

        Parameters
        ----------
        inputs : ArrayLike
            Inputs. [B, L, dm]
        with_dropout : bool, optional
            Whether dropout or not

        Returns
        -------
        inputs : Array
           Encoded Inputs. [B, L, dm]
        """
        B, L, dm = inputs.shape

        mha = MultiHeadProbSparseAttention(
            c=self.c,
            nH=self.nH,
            dm=self.dm,
            Pdrop=self.Pdrop,
            mask=False
        )
        ff = FeedForward(dff=self.dff, Pdrop=self.Pdrop)

        inputs = ResidualLayerNorm(
            lambda i: mha(i, i, i, with_dropout=with_dropout),
            self.eps
        )(inputs)
        inputs = inputs.at[:].set(
            ResidualLayerNorm(
                lambda i: ff(i, with_dropout=with_dropout),
                self.eps
            )(inputs)
        )

        assert inputs.shape == (B, L, dm), "BUG"
        return inputs


class DecoderLayer(nn.Module):
    """
    Decoder Layer
    """
    c: int
    dm: int
    Ltoken: int
    nH: int = NH
    dff: int = DFF
    eps: float = EPS
    Pdrop: float = PDROP


    @nn.compact
    def __call__(self,
                 inputs: ArrayLike,
                 outputs: ArrayLike, *,
                 with_dropout: bool = False) -> Array:
        """
        Call Decoder Layer

        Parameters
        ----------
        inputs : ArrayLike
            Inputs. [B, Lenc, dm]
        outputs : ArrayLike
            Outputs. [B, Ldec, dm]
        with_dropout : bool, optional
            Whether dropout or not

        Returns
        -------
        outputs : Array
           Decoded Outputs. [B, L, dm]
        """
        B, Ldec, dm = outputs.shape

        mmha = MultiHeadProbSparseAttention(
            c=self.c,
            nH=self.nH,
            dm=self.dm,
            Pdrop=self.Pdrop,
            mask=self.Ltoken,
            name="MaskedMultiHeadProbSparseAttention",
        )
        mha = MultiHeadAttention(
            nH=self.nH,
            dm=self.dm,
            Pdrop=self.Pdrop,
            name="MultiHeadAttention",
        )
        ff = FeedForward(dff=self.dff, Pdrop=self.Pdrop)

        outputs = ResidualLayerNorm(
            lambda o: mmha(o, o, o, with_dropout=with_dropout),
            self.eps
        )(outputs)

        outputs = outputs.at[:].set(ResidualLayerNorm(
            lambda o: mha(o, inputs, inputs, with_dropout=with_dropout),
            self.eps
        )(outputs))

        outputs = outputs.at[:].set(
            ResidualLayerNorm(
                lambda o: ff(o, with_dropout=with_dropout),
                self.eps
            )(outputs)
        )

        assert outputs.shape == (B, Ldec, dm), "BUG"
        return outputs


class EncoderStack(nn.Module):
    """
    Encoder Stack
    """
    c: int
    dm: int
    nE: int = NE
    nH: int = NH
    dff: int = DFF
    kernel: int = KERNEL_SIZE
    eps: float = EPS
    Pdrop: float = PDROP

    @nn.compact
    def __call__(self,
                 inputs: ArrayLike, *,
                 with_dropout: bool = False) -> Array:
        """
        Call Encoder Stack

        Parameters
        ----------
        inputs : ArrayLike
            Inputs. [B, L, dm]
        with_dropout : bool, optional
            Whether dropout or not.

        Returns
        -------
        inputs : Array
            Encoded Inputs. [B, L, dm]
        """
        for i in range(self.nE):
            B, L, dm = inputs.shape
            inputs = EncoderLayer(c=self.c,
                                  dm=self.dm,
                                  nH=self.nH,
                                  dff=self.dff,
                                  eps=self.eps,
                                  Pdrop=self.Pdrop,
                                  name=f"EncoderLayer_{i}")(inputs,
                                                            with_dropout=with_dropout)
            assert inputs.shape == (B, L, dm), "BUG"

            inputs = Distilling(kernel=self.kernel,
                                name=f"DistillingLayer_{i}")(inputs)
            assert inputs.shape == (B, (L+1)//2, dm), "BUG"

        return nn.LayerNorm(epsilon=self.eps)(inputs)


class DecoderStack(nn.Module):
    """
    Decoder Stack
    """
    c: int
    dm: int
    Ltoken: int
    nD: int = ND
    nH: int = NH
    dff: int = DFF
    eps: float = EPS
    Pdrop: float = PDROP

    @nn.compact
    def __call__(self,
                 inputs: ArrayLike,
                 outputs: ArrayLike, *,
                 with_dropout: bool = False) -> Array:
        """
        Call Encoder Stack

        Parameters
        ----------
        inputs : ArrayLike
            Encoded Inputs. [B, Lenc, dm]
        outputs : ArrayLike
            Outputs. [B, Ldec, dm]
        with_dropout : bool, optional
            Whether dropout or not.

        Returns
        -------
        outputs : Array
            Decoded Outputs. [B, Ldec, dm]
        """
        B, L, dm = outputs.shape

        for i in range(self.nD):
            outputs = DecoderLayer(c=self.c,
                                   dm=self.dm,
                                   Ltoken=self.Ltoken,
                                   nH=self.nH,
                                   dff=self.dff,
                                   eps=self.eps,
                                   Pdrop=self.Pdrop,
                                   name=f"DecoderLayer_{i}")(
                                       inputs,
                                       outputs,
                                       with_dropout=with_dropout
                                   )
            assert outputs.shape == (B, L, dm), "BUG"

        return nn.LayerNorm(epsilon=self.eps)(outputs)


class Informer(Model):
    """
    Informer

    Attributes
    ----------
    d : int
        Dimension of Sequence Data
    I : int
        Input Length (aka. Lookback Horizon)
    O : int
        Output Length (aka. Prediction Horizon)
    Ltoken : int
        Length of Start Token for Decoder
    c : int
        Hyper Parameter of Sampling Factor
    dm : int
        Model Dimension
    Vs : tuple of ints, optional
        Dimensions of Categorical Features
    alpha : float
        Rescale Facotor after Embedding. If input sequence is normalized,
        ``alpha=1.0`` is enough.
    nE : int, optional
        Number of Encoder Layers
    nD : int, optional
        Number of Decoder Layers
    nH : int
        Number of Multi Head
    dff : int, optional
        Hidden Layer Units at Feed Forward Layer
    kernel : int, optional
        Kernel Size for Distilling Layer
    eps : float, optional
        Small Positive Value for Layer Normalization
    Pdrop : foat, optional
        Dropout Probability
    """
    d: int
    I: int
    O: int
    Ltoken: int
    c: int
    dm: int
    Vs: Tuple[int, ...] = tuple()
    alpha: float = EMBEDDING_ALPHA
    nE: int = NE
    nD: int = ND
    nH: int = NH
    dff: int = DFF
    kernel: int = KERNEL_SIZE
    eps: float = EPS
    Pdrop: float = PDROP

    def setup(self):
        assert self.I >= self.Ltoken, "BUG"

        self.encoder = EncoderStack(c=self.c,
                                    nE=self.nE,
                                    dm=self.dm,
                                    nH=self.nH,
                                    dff=self.dff,
                                    kernel=self.kernel,
                                    eps=self.eps,
                                    Pdrop=self.Pdrop)
        self.encoder_embed = Embedding(dm=self.dm,
                                       Vs=self.Vs,
                                       alpha=self.alpha,
                                       kernel=self.kernel,
                                       Pdrop=self.Pdrop)

        self.decoder = DecoderStack(c=self.c,
                                    nD=self.nD,
                                    dm=self.dm,
                                    Ltoken=self.Ltoken,
                                    nH=self.nH,
                                    dff=self.dff,
                                    eps=self.eps,
                                    Pdrop=self.Pdrop)
        self.decoder_embed = Embedding(dm=self.dm,
                                       Vs=self.Vs,
                                       alpha=self.alpha,
                                       kernel=self.kernel,
                                       Pdrop=self.Pdrop)
        self.ff = nn.Dense(features=self.d)

    def encode(self,
               seq: ArrayLike,
               cat: Optional[ArrayLike] = None, *,
               with_dropout: bool = False) -> Array:
        """
        Encode with Informer

        Paremeters
        ----------
        seq : ArrayLike
            Inputs. [B, I, d]
        cat : ArrayLike, optional
            Categorical Features. [B, I, C]
        with_dropout : bool
            Whether dropout or not

        Returns
        -------
        inputs : Array
            Encoded Inputs. [B, L, dm]
        """
        assert (cat is None) or seq.shape[:1] == cat.shape[:1], "BUG"
        assert seq.shape[1] == self.I, "BUG"

        B = seq.shape[0]

        inputs = self.encoder_embed(seq, cat, with_dropout=with_dropout)
        assert inputs.shape == (B, self.I, self.dm), "BUG"

        inputs = self.encoder(inputs, with_dropout=with_dropout)
        assert inputs.shape[0] == B, "BUG"
        assert inputs.shape[2] == self.dm, "BUG"

        return inputs

    def decode(self,
               inputs: ArrayLike,
               seq: ArrayLike,
               cat: Optional[ArrayLike] = None, *,
               with_dropout: bool = False) -> Array:
        """
        Decode with Informer

        Parameters
        ----------
        inputs : ArrayLike
            Encoded Inputs. [B, L, dm]
        seq : ArrayLike
            Outputs Signal. [B, L, d]
        cat : AllayLike, optional
            Categorical Features. [B, L, d]
        """
        assert inputs.shape[0] == seq.shape[0], "BUG"
        assert inputs.shape[2] == self.dm, "BUG"
        assert seq.shape[1] >= self.Ltoken, "BUG"
        assert ((cat is None) or seq.shape[:2] == cat.shape[:2]), "BUG"
        assert ((cat is None) or cat.shape[2] == len(self.Vs))

        B = inputs.shape[0]

        seq = seq.at[:,-self.Ltoken:,:].get()
        if cat is not None:
            cat = cat.at[:,-self.Ltoken:,:].get()

        outputs = (jnp.zeros((B, self.Ltoken+self.O, self.dm), dtype=seq.dtype)
                   .at[:,:self.Ltoken,:]
                   .set(self.decoder_embed(seq, cat, with_dropout=with_dropout)))
        assert outputs.shape == (B, self.Ltoken+self.O, self.dm), "BUG"

        outputs = outputs.at[:].set(
            self.decoder(inputs, outputs, with_dropout=with_dropout)
        )

        outputs = self.ff(outputs)
        assert outputs.shape == (B, self.Ltoken+self.O, self.d), "BUG"

        pred = outputs.at[:,-self.O:,:].get()
        return pred

    def __call__(self,
                 seq: ArrayLike,
                 cat: Optional[ArrayLike] = None, *,
                 train: bool = False) -> Array:
        """
        Call Informer

        Parameters
        ----------
        seq : ArrayLike
            Inputs Signal. [B, I, d]
        cat : ArrayLike, optional
            Categorical Features. [B, I, C]
        train : bool, optional
            Whether dropout or not.

        Returns
        -------
        pred : Array
            Predicted Signal. [B, O, dm]
        """
        assert (cat is None) or seq.shape[:2] == cat.shape[:2], "BUG"
        assert seq.shape[1:] == (self.I, self.d), f"BUG: {seq.shape}"
        assert (cat is None) or cat.shape[2] == len(self.Vs), "BUG"

        B: int = seq.shape[0]

        inputs = self.encode(seq, cat, with_dropout=train)
        assert inputs.shape[0] == B, "BUG"
        assert inputs.shape[1] <= self.I
        assert inputs.shape[2] == self.dm, "BUG"

        pred = self.decode(inputs, seq, cat, with_dropout=train)
        assert pred.shape == (B, self.O, self.d), "BUG"

        return pred

    @staticmethod
    def split_key(key: KeyArray, *,
                  train: bool = False) -> Tuple[KeyArray, Dict[str, KeyArray]]:
        """
        Split PRNG Key for Informer

        Parameters
        ----------
        key : KeyArray
            Key will be split
        train : bool, optional
            Whether train or not

        Returns
        -------
        key : KeyArray
            New Key
        key_for_model : KeyArray
            Keys can be consumed by Informer.
        """
        if train:
            key, key_a, key_d = jax.random.split(key, 3)
            return key, {"attention": key_a, "dropout": key_d}

        key, key_a = jax.random.split(key, 2)
        return key, {"attention": key_a}

    def log_model(self) -> None:
        """
        Log Informer Spec
        """
        logger.info("Informer(d=%d, I=%d, O=%d, Ltoken=%d, c=%d, dm=%d,"
                    " Vs=%s, alpha=%d, nE=%d, nD=%d, nH=%d, dff=%d,"
                    " kernel=%d, eps=%.2e, Pdrop=%f)",
                    self.d, self.I, self.O, self.Ltoken, self.c, self.dm,
                    self.Vs, self.alpha, self.nE, self.nD, self.nH, self.dff,
                    self.kernel, self.eps, self.Pdrop)
