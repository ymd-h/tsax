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
import functools
from typing import Dict, Optional, Tuple
import math

import jax
import jax.numpy as jnp
from jax.random import KeyArray
from flax import linen as nn
import wblog

from tsax.typing import Array
from tsax.core import (
    Model,
    ConvSeq,
    FeedForward,
    Embedding,
    ResidualLayerNorm,
    SubsequentMask,
    MultiHeadAttention,
)
from tsax.core.encoding import EMBEDDING_ALPHA

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
]


logger = wblog.getLogger()


KERNEL_SIZE: int = 3
"""
Default Kernel Size for Conv1d at Distilling Layer
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


class Distilling(nn.Module):
    """
    Distilling Layer
    """
    kernel: int = KERNEL_SIZE

    @nn.compact
    def __call__(self, x: Array) -> Array:
        """
        Call Distilling Layer

        Parameters
        ----------
        x : Array
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
                 Q: Array,
                 K: Array,
                 V: Array) -> Array:
        """
        Call Attention Layer

        Parameters
        ----------
        Q : Array
            Query. [B, Lq, dm]
        K : Array
            Key. [B, Lk, dm]
        V : Array
            Value. [B, Lk, dm]

        Returns
        -------
        A : Array
            Attention. [B, Lk, dv]
        """
        assert K.shape == V.shape, "BUG"
        assert Q.shape[0] == K.shape[0], "BUG"
        assert Q.shape[2] == K.shape[2], "BUG"

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
    mask: bool = False
    rng_collection: str = "attention"

    @nn.compact
    def __call__(self,
                 Q: Array,
                 K: Array,
                 V: Array) -> Array:
        """
        Call ProbSparse self-attention

        Parameters
        ----------
        Q : Array
            Query. [B, LQ, dm]
        K : Array
            Key. [B, LK, dm]
        V : Array
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

        if self.mask:
            mask = SubsequentMask(n)

        @jax.vmap
        def _each_sample(_Q, _K, _V, _rng):
            _Kbar = _K.at[jax.random.choice(_rng,
                                            _K.shape[0],
                                            shape=(U,),
                                            replace=False),:].get()
            assert _Kbar.shape == (U, self.dk), "BUG"

            _Sbar = jnp.tensordot(_Q, _Kbar, axes=((1,), (1,)))
            assert _Sbar.shape == (m, U), "BUG"

            _M = jnp.max(_Sbar, axis=1) - jnp.mean(_Sbar, axis=1)
            assert _M.shape == (m,), f"BUG: {_M.shape} != {(m,)}"

            _, _I = jax.lax.top_k(_M, u)
            assert _I.shape == (u,), "BUG"

            _Qbar = _Q.at[_I, :].get()
            assert _Qbar.shape == (u, self.dk), "BUG"

            _QbarK = jnp.tensordot(_Qbar, _K, axes=((1,), (1,)))
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
                 inputs: Array, *,
                 with_dropout: bool = False) -> Array:
        """
        Call Encoder Layer

        Parameters
        ----------
        inputs : Array
            Inputs. [B, L, dm]
        with_dropout : bool, optional
            Whether dropout or not

        Returns
        -------
        inputs : Array
           Encoded Inputs. [B, L, dm]
        """
        B, L, dm = inputs.shape

        mha = MultiHeadAttention(
            attention=functools.partial(ProbSparseAttention, c=self.c, mask=False),
            dm=self.dm,
            nH=self.nH,
            Pdrop=self.Pdrop,
        )
        ff = FeedForward(dff=self.dff, Pdrop=self.Pdrop, activation="GELU")

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
    nH: int = NH
    dff: int = DFF
    eps: float = EPS
    Pdrop: float = PDROP


    @nn.compact
    def __call__(self,
                 inputs: Array,
                 outputs: Array, *,
                 with_dropout: bool = False) -> Array:
        """
        Call Decoder Layer

        Parameters
        ----------
        inputs : Array
            Inputs. [B, Lenc, dm]
        outputs : Array
            Outputs. [B, Ldec, dm]
        with_dropout : bool, optional
            Whether dropout or not

        Returns
        -------
        outputs : Array
           Decoded Outputs. [B, L, dm]
        """
        B, Ldec, dm = outputs.shape

        mmha = MultiHeadAttention(
            attention=functools.partial(ProbSparseAttention,
                                        c=self.c,
                                        mask=True),
            dm=self.dm,
            nH=self.nH,
            Pdrop=self.Pdrop,
        )
        mha = MultiHeadAttention(
            attention=Attention,
            dm=self.dm,
            nH=self.nH,
            Pdrop=self.Pdrop,
        )
        ff = FeedForward(dff=self.dff, Pdrop=self.Pdrop, activation="GELU")

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
                 inputs: Array, *,
                 with_dropout: bool = False) -> Array:
        """
        Call Encoder Stack

        Parameters
        ----------
        inputs : Array
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

            if i < self.nE - 1:
                # Last Layer doesn't have following Distilling Layer.
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
    nD: int = ND
    nH: int = NH
    dff: int = DFF
    eps: float = EPS
    Pdrop: float = PDROP

    @nn.compact
    def __call__(self,
                 inputs: Array,
                 outputs: Array, *,
                 with_dropout: bool = False) -> Array:
        """
        Call Encoder Stack

        Parameters
        ----------
        inputs : Array
            Encoded Inputs. [B, Lenc, dm]
        outputs : Array
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
                                       Pdrop=self.Pdrop,
                                       with_positional=True)

        self.decoder = DecoderStack(c=self.c,
                                    nD=self.nD,
                                    dm=self.dm,
                                    nH=self.nH,
                                    dff=self.dff,
                                    eps=self.eps,
                                    Pdrop=self.Pdrop)
        self.decoder_embed = Embedding(dm=self.dm,
                                       Vs=self.Vs,
                                       alpha=self.alpha,
                                       kernel=self.kernel,
                                       Pdrop=self.Pdrop,
                                       with_positional=True)
        self.ff = nn.Dense(features=self.d)

    def encode(self,
               seq: Array,
               cat: Optional[Array] = None, *,
               with_dropout: bool = False) -> Array:
        """
        Encode with Informer

        Paremeters
        ----------
        seq : Array
            Inputs. [B, I, d]
        cat : Array, optional
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
               inputs: Array,
               seq: Array,
               cat: Optional[Array] = None, *,
               with_dropout: bool = False) -> Array:
        """
        Decode with Informer

        Parameters
        ----------
        inputs : Array
            Encoded Inputs. [B, L, dm]
        seq : Array
            Outputs Signal. [B, L, d]
        cat : AllayLike, optional
            Categorical Features. [B, L, d]

        Returns
        -------
        pred : Array
            Predicted Signal. [B, O, d]
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
                 seq: Array,
                 cat: Optional[Array] = None, *,
                 train: bool = False) -> Array:
        """
        Call Informer

        Parameters
        ----------
        seq : Array
            Inputs Signal. [B, I, d]
        cat : Array, optional
            Categorical Features. [B, I, C]
        train : bool, optional
            Whether train or not.

        Returns
        -------
        pred : Array
            Predicted Signal. [B, O, d]
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
                    " Vs=%s, alpha=%f, nE=%d, nD=%d, nH=%d, dff=%d,"
                    " kernel=%d, eps=%.2e, Pdrop=%f)",
                    self.d, self.I, self.O, self.Ltoken, self.c, self.dm,
                    self.Vs, self.alpha, self.nE, self.nD, self.nH, self.dff,
                    self.kernel, self.eps, self.Pdrop)
