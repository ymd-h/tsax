"""
Autoformer (:mod:`tsax.autoformer`)
===================================

Notes
-----
This module implements Autoformer [1]_.

References
----------
.. [1] H. Wu et al., "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting", NeurIPS 2021,
   https://proceedings.neurips.cc/paper_files/paper/2021/hash/bcc0d400288793e8bdcd7c19a8ac0c2b-Abstract.html,
   https://arxiv.org/abs/2106.13008
"""
from __future__ import annotations
import math
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from flax import linen as nn

from tsax.core import (
    Model,
    ConvSeq,
    Embedding,
    FeedForward,
)
from tsax.core.encoding import EMBEDDING_ALPHA

__all__ = [
    "Autoformer",
    "EncodeStack",
    "DecoderStack",
    "EncoderLayer",
    "DecoderLayer",
    "MutiHeadAttention",
    "AutoCorrelationAttention",
    "SeriesDecomp",
]

K_MOVING_AVG: int = 25
"""
Default Length of Moving Average
"""

DFF: int = 2048
"""
Default Number of Units at Hidden Layer of Feed Forward
"""

NH: int = 8
"""
Default Number of Multi Head Attention
"""

NE: int = 2
"""
Default Number of Encoder Stack
"""

ND: int = 1
"""
Default Number of Decoder Stack
"""


EPS: float = 1e-12
"""
Default Value for Small Positive Value of Layer Normalization
"""

PDROP: float = 0.1
"""
Default Probability of Dropout Rate
"""


class SeriesDecomp(nn.Module):
    """
    Series Decomposition

    Attributes
    ----------
    kMA : int
        Window Size of Moving Average
    """
    kMA: int = K_MOVING_AVG

    @nn.compact
    def __call__(self, x: ArrayLike) -> Tuple[Array, Array]:
        """
        Call Series Decomposition

        Parameters
        ----------
        x : ArrayLike
            Input Series. [B, L, d]

        Returns
        -------
        season : Array
            Seasonal Component. [B, L, d]
        trend : Array
            Trend Component. [B, L, d]
        """
        L = x.shape[1]
        left = kMA // 2
        right = kMA - left

        trend = jax.vmap(
            lambda idx: x.at[:, idx-left:idx+right].get(mode="clip").mean(axis=0)
        )(jnp.arange(L))
        assert x.shape == trend.shape

        season = x - trend
        return season, trend


class SeasonalLayerNorm(nn.Module):
    """
    Layer Normalization for Seasonal

    Attributes
    ----------
    eps : float
        Small Positive Value for LayerNorm
    """
    eps: float

    @nn.compact
    def __call__(self, x: ArrayLike) -> Array:
        """
        Call Seasonal Layer Normalization

        Parameters
        ----------
        x : ArrayLike
            Input Sequence. [B, L, d]

        Returns
        -------
        x : Array
            Output Sequcence. [B, L, d]
        """
        x = nn.LayerNorm(epsilon=self.eps)(x)
        x = x.at[:].sub(jnp.mean(x, axis=1, keepdims=True))

        return  x


class AutoCorrelationAttention(nn.Module):
    """
    AutoCorrelation Attention

    Attributes
    ----------
    d : int
        Query, Key, Value Dimension
    c : int
        Coefficient for Selecting Top K Correlation.
        floor(c * logL) Correlation is used.
        ``1 <= c <= 3``
    """
    d: int
    c: int

    @nn.compact
    def __call__(self,
                 Q: ArrayLike,
                 K: ArrayLike,
                 V: ArrayLike) -> Array:
        """
        Call AutoCorrelation Attention Layer

        Paremeters
        ----------
        Q : ArrayLike
            Query. [B, L, dm]
        K : ArrayLike
            Key. [B, S, dm]
        V : ArrayLike
            Value. [B, S, dm]

        Returns
        -------
        A : Array
            Auto-Correlation as Attention. [B, L, d]
        """
        assert Q.shape[0] == K.shape[0] == V.shape[0], "BUG"
        assert K.shape[1] == V.shape[1], "BUG"
        assert Q.shape[2] == K.shape[0] == V.shape[2], "BUG"

        B, L, _ = Q.shape

        # Resize by truncation or 0 filling.
        # K, V: [B, S, dm] -> [B, L, dm]
        K = K.at[:,0:L,:].get(mode="fill", fill_value=0)
        V = V.at[:,0:L,:].get(mode="fill", fill_value=0)

        # Q, K: [B, L, dm] -> [B, L, dk]
        Q = nn.Dense(features=self.d, name="WQ")(Q)
        K = nn.Dense(features=self.d, name="WK")(K)

        # V: [B, L, dm] -> [B, L, dv]
        V = nn.Dense(features=self.d, name="WV")(V)

        # Q_freq, K_freq: [B, L, dk]
        Q_freq = jnp.fft.rfft(Q, axis=1)
        K_freq = jnp.fft.rfft(K, axis=1)
        assert Q_freq.shape == K_freq.shape == (B, L, self.d), "BUG"

        # Rxx: [B, L, dk]
        Rxx = jnp.fft.irfft(Q_freq * jnp.conjugate(K_freq), n=L, axis=1)
        assert Rxx.shape == (B, L, self.d), "BUG"

        # Time Delay Aggregation
        # ----------------------

        # Note: Use ``math`` for static calculation
        k = int(math.floor(self.c * math.log(L)))

        Wk, Ik = jax.lax.top_k(jnp.moveaxis(Rxx, 1, -1), k)
        assert Wk.shape == Ik.shape == (B, self.d, k), "BUG"

        Wk = flax.activation.softmax(Wk, axis=-1)

        @jax.vmap
        def f(_w, _i, _v):
            assert _v.shape == (L,), "BUG"
            f_ret =  _w * jnp.roll(_v, -_i)
            assert f_ret.shape == _v.shape, "BUG"
            return f_ret

        @jax.vmap
        def g(_wk, _ik, _V):
            assert _wk.shape == _ik.shape == (self.d,), "BUG"
            g_ret = f(_wk, _ik, _V)
            assert g_ret.shape == (self.d, L), "BUG"
            return g_ret

        @jax.vmap
        def h(wk, ik):
            assert wk.shape == ik.shape == (B, self.d), "BUG"
            h_ret = g(wk, ik, jnp.moveaxis(V, 1, -1))
            assert h_ret.shape == (B, self.d, L), "BUG"
            return jnp.moveaxis(h_ret, -1, 1)

        # A: [B, L, dk]
        A = jnp.sum(h(jnp.moveaxis(Wk, -1, 0), jnp.moveaxis(Ik, -1, 0)), axis=0)
        assert A.shape == (B, L, self.d), "BUG"

        return A


class MultiHeadAttention(nn.Module):
    """
    Multi HEad Attention Layer

    Attributes
    ----------
    dm : int
        Model Dimension
    nH : int
        Number of Multi Head
    Pdrop : float
        Dropout Rate
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
        Multi Head Attention

        Parameters
        ----------
        Q : ArrayLike
            Query. [B, L, dm]
        K : ArrayLike
            Key. [B, L, dm]
        V : ArrayLike
            Value. [B, L, dm]
        with_dropout : bool, optional
            Whether dropout or not.

        Returns
        -------
        MHA : Array
            Multi Head Attention. [B, L, dm]
        """
        assert Q.shape[0] == K.shape[0] == V.shape[0], "BUG"
        assert K.shape[1] == V.shape[1], "BUG"
        assert Q.shape[2] == K.shape[2] == V.shape[2], "BUG"

        # x: [B, L, dm (= dm/nH * nH)]
        d: int = self.dm // self.nH
        x = jnp.concatenate([AutoCorrelationAttention(d=d, name=f"head_{i}")(Q, K, V)
                             for i in range(self.nH)],
                            axis=2)
        assert x.shape == (*Q.shape[:2], d * self.nH)

        # MHA: [B, L, dm]
        MHA = nn.Dense(features=self.dm, name="WO")
        assert Q.shape == MHA.shape, "BUG"

        if with_dropout:
            MHA = MHA.at[:].set(nn.Dropout(self.Pdrop, deterministic=False)(MHA))

        return MHA


class EncoderLayer(nn.Module):
    """
    Encoder Layer

    Attributes
    ----------
    dm : int
        Model Dimension
    nH : int
        Number of Multi Head
    kMA : int
        Window Size of Moving Average
    eps : float
        Small Positive Value for Layer Normalization
    Pdrop : float
        Dropout Rate
    """
    dm: int
    nH: int = NH
    kMA: int = K_MOVING_AVG
    dff: int = DFF
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
            Whether dropout or not.

        Returns
        -------
        inputs : Array
            Encoded Inputs. [B, L, dm]
        """
        shape = inputs.shape

        mha = MultiHeadAttention(nH=self.nH, dm=self.dm, Pdrop=self.Pdrop)
        ff = FeedForward(dff=self.dff, Pdrop=self.Pdrop, bias=False)

        inputs, _ = SeriesDecomp(kMA=self.kMA)(
            inputs + mha(inputs, inputs, inputs, with_dropout=with_dropout)
        )

        inputs, _ = SeriesDecomp(kMA=self.kMA)(
            inputs + ff(inputs, with_dropout=with_dropout)
        )

        assert inputs.shape == shape, "BUG"
        return inputs


class DecoderLayer(nn.Module):
    """
    Decoder Layer

    Attributes
    ----------
    dm : int
        Model Dimension
    nH : int
        Number of Multi Head
    dff : int
        Number of Hidden Units at Feed Forward
    kMA : int
        Window Size of Moving Average
    eps : float
        Small Positive Value for Layer Normalization
    Pdrop : float
        Dropout Rate
    """
    dm: int
    nH: int = NH
    dff: int = DFF
    kMA: int = K_MOVING_AVG
    Pdrop: float = PDROP

    @nn.compact
    def __call__(self,
                 inputs: ArrayLike,
                 seasonal_outputs: ArrayLike,
                 trend_outputs: ArrayLike, *,
                 with_dropout: bool = False) -> Tuple[Array, Array]:
        """
        Call Decloder Layer

        Parameters
        ----------
        inputs : ArrayLike
            Encoded Inputs. [B, S, dm]
        seasonal_outputs : ArrayLike
            Seasonal Outputs. [B, L, dm]
        trend_outputs : ArrayLike
            Trend-Cyclical Outputs. [B, L, dm]
        with_dropout : bool, optional
            Whether dropout or not.

        Returns
        -------
        seasonal_outputs : Array
            Seasonal Outputs. [B, L, dm]
        trend_outputs : Array
            Trend Outputs. [B, L, d]
        """
        assert seasonal_outputs.shape == trend_outputs.shape, "BUG"
        assert inputs.shape[0] == seasonal_outputs.shape[0], "BUG"
        assert inputs.shape[2] == seasonal_outputs.shape[2] == self.dm, "BUG"

        s_mha = MultiHeadAttention(nH=self.nH, dm=self.dm, Pdrop=self.Pdrop,
                                   name="SelfAttention")
        c_mha = MultiHeadAttention(nH=self.nH, dm=self.dm, Pdrop=self.Pdrop,
                                   name="CrossAttention")
        ff = FeedForward(dff=self.dff, Pdrop=self.Pdrop, bias=False)

        seasonal_outputs, trend1 = SeriesDecomp(kMA=self.kMA)(
            seasonal_outputs + s_mha(seasonal_outputs,
                                     seasonal_outputs,
                                     seasonal_outputs,
                                     with_dropout=with_dropout)
        )

        seasonal_outputs, trend2 = SeriesDecomp(kMA=self.kMA)(
            seasonal_outputs + c_mha(seasonal_outputs,
                                     inputs,
                                     inputs,
                                     with_dropout=with_dropout)
        )

        seasonal_outputs, trend3 = SeriesDecomp(kMA=self.kMA)(
            seasonal_outputs + ff(seasonal_outputs)
        )

        trend_outputs.at[:].add(
            ConvSeq(dm=self.dm, kernel=3, bias=False)(trend1 + trend2 + trend3)
        )

        return seasonal_outputs, trend_outputs


class EncoderStack(nn.Module):
    """
    Encoder Stack

    Attributes
    ----------
    dm : int
        Model Dimension
    nE : int
        Number of Encoder Layers
    nH : int
        Number of Multi Head
    dff : int
        Number of Hidden Units at Feed Forward
    kMA : int
        Window Size of Moving Average
    eps : float
        Small Positive Value for Layer Normalization
    Pdrop : float
        Dropout Rate
    """
    dm: int
    nE: int = NE
    nH: int = NH
    dff: int = DFF
    kMA: int = K_MOVING_AVG
    eps: float = EPS
    Pdrop: float = PDROP

    @nn.compact
    def __call__(self,
                 inputs: ArrayLike,
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
        assert inputs.shape[2] == self.dm, "BUG"

        shape = inputs.shape

        for i in range(self.N):
            inputs = EncoderLayer(nH=self.nH,
                                  dm=self.dm,
                                  dff=self.dff,
                                  kMA=self.kMA,
                                  Pdrop=self.Pdrop,
                                  name=f"EncoderLayer_{i}")(inputs,
                                                            with_dropout=with_dropout)

        inputs = inputs.at[:].set(SeasonalLayerNorm(eps=self.eps)(inputs))

        assert inputs.shape == shape, "BUG"
        return inputs


class DecoderStack(nn.Module):
    """
    Decoder Stack

    Attributes
    ----------
    dm : int
        Model Dimension
    N : int
        Number of Decoder Layers
    nH : int
        Number of Multi Head
    dff : int
        Number of Hidden Units at Feed Forward
    kMA : int
        Window Size of Moving Average
    eps : float
        Small Positive Value for Layer Normalization
    Pdrop : float
        Dropout Rate
    """
    dm: int
    nD: int = ND
    nH: int = NH
    dff: int = DFF
    kMA: int = K_MOVING_AVG
    eps: float = EPS
    Pdrop: float = PDROP

    @nn.compact
    def __call__(self,
                 inputs: ArrayLike,
                 sesonal_outputs: ArrayLike,
                 trend_outputs: ArrayLike, *,
                 with_dropout: bool = False) -> Tuple[Array, Array]:
        """
        Call Decoder Stack

        Parameters
        ----------
        inputs : ArrayLike
            Encoded Inputs. [B, S, dm]
        seasonal_outputs : ArrayLike
            Seasonal Outputs. [B, L, dm]
        trend_outputs : ArrayLike
            Trend-Cyclical Outputs. [B, L, dm]
        with_dropout : bool
            Whether dropout or not.

        Returns
        -------
        seasonal_outputs : Array
            Seasonal Outputs. [B, L, dm]
        trend_outputs : Array
            Trend-Cyclical Outputs. [B, L, dm]
        """
        assert seasonal_outputs.shape == trend_outputs.shape, "BUG"
        assert inputs.shape[0] == seasonal_outputs.shape[0], "BUG"
        assert inputs.shape[2] == seasonal_outputs.shape[2] == self.dm, "BUG"

        for i in range(self.nD):
            seasonal_outputs, trend_outputs = DecoderLayer(
                nH=self.nH,
                dm=self.dm,
                dff=self.dff,
                kMA=self.kMA,
                Pdrop=self.Pdrop,
                name=f"DecoderLayer_{i}"
            )(
                inputs,
                seasonal_outputs,
                trend_outputs,
                with_dropout=with_dropout
            )

        seasonal_outputs = seasonal_outputs.at[:].set(
            SeasonalLayerNorm(eps=self.eps)(seasonal_outputs)
        )

        seasonal_outputs = nn.Dense(features=self.dm)(seasonal_outputs)
        assert seasonal_outputs.shape == trend_outputs.shape, "BUG"

        return seasonal_outputs, trend_outputs


class Autoformer(Model):
    """
    Autoformer

    Attributes
    ----------
    """
    d: int
    I: int
    O: int
    Vs: Tuple[int, ...] = tuple()
    alpha: float = EMBEDDING_ALPHA
    nE: int = NE
    nD: int = ND
    nH: int = NH
    dff: int = DFF
    kMA: int = K_MOVING_AVG
    eps: float = EPS
    Pdrop: float = PDROP

    def setup(self):
        self.encoder = EncoderStack(N=self.nE,
                                    dm=self.d,
                                    nH=self.nH,
                                    dff=self.dff,
                                    kMA=self.kMA,
                                    eps=self.eps,
                                    Pdrop=self.Pdrop)
        self.encoder_embed = Embedding(dm=self.d,
                                       Vs=self.Vs,
                                       kernel=3,
                                       alpha=self.alpha,
                                       Pdrop=self.Pdrop,
                                       with_positional=False)

        self.decoder = DecoderStack(N=self.nD,
                                    dm=self.d,
                                    nH=self.nH,
                                    dff=self.dff,
                                    kMA=self.kMA,
                                    eps=self.eps,
                                    Pdrop=self.Pdrop)
        self.decoder_embed = Embedding(dm=self.d,
                                       Vs=self.Vs,
                                       kernel=3,
                                       alpha=self.alpha,
                                       Pdrop=self.Pdrop,
                                       with_positional=False)
        self.ff = nn.Dense(features=self.d)

    def encode(self,
               seq: ArrayLike,
               cat: Optional[ArrayLike] = None, *,
               with_dropout: bool = False) -> Array:
        """
        Encode with Autoformer

        Parameters
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
            Encoded Inputs. [B, I, d]
        """
        assert (cat is None) or seq.shape[:1] == cat.shape[:1], "BUG"
        assert seq.shape[1] == self.I, "BUG"

        B = inputs.shape[0]

        inputs = self.encoder_embed(seq, cat, with_dropout=with_dropout)
        assert inputs.shape == (B, self.I, self.d), "BUG"

        inputs = self.encoder(inputs, with_dropout=with_dropout)
        assert inputs.shape == (B, self.I, self.d), "BUG"

        return inputs

    def decode(self,
               inputs: ArrayLike,
               seq: ArrayLike,
               cat: Optional[ArrayLike] = None, *,
               with_dropout: bool = False) -> Array:
        """
        Decode with Autoformer

        Parameters
        ----------
        inputs : ArrayLike
            Encoded Inputs. [B, L, d]
        seq : ArrayLike
            Outputs Signal. [B, L, d]
        cat : AllayLike, optional
            Categorical Features. [B, L, d]

        Returns
        -------
        pred : Array
            Predicted Signal. [B, O, d]
        """
        assert inputs.shape[0] == seasonal_outputs.shape[0], "BUG"
        assert inputs.shape[2] == seasonal_outputs.shape[2], "BUG"
        assert (cat is None) or (seq.shape[:2] == cat.shape[:2]), "BUG"
        assert (cat is None) or (cat.shape[2] == len(self.Vs)), "BUG"

        B: int = inputs.shape[0]
        S: int = self.I // 2
        L: int = S + self.O


        s, t = SeriesDecomp(seq.at[:,:S,:].get())

        s_outputs = jnp.zeros((B, L, self.d), dtype=seq.dtype).at[:,:S,:].set(s)
        t_outputs = (jnp.zeros((B, L, self.d), dtype=seq.dtype)
                     .at[:,:S,:].set(t)
                     .at[:,S:,:].set(jnp.mean(seq, axis=1, keepdims=True)))

        # Only seasonal part is embedded.
        s_outputs = self.decoder_embed(s_outputs, cat, with_dropout=with_dropout)
        assert s_outputs.shape == (B, L, self.d)

        s_outputs, t_outputs = self.decoder(inputs, s_outputs, t_outputs,
                                            with_dropout=with_dropout)
        assert s_outputs.shape == t_outputs.shape == (B, L, self.d), "BUG"

        pred = s_outputs.at[:,L-O:,:].get() + t_outputs.at[:,L-O:,:].get()
        return pred

    def __call__(self,
                 seq: ArrayLike,
                 cat: Optional[ArrayLike] = None, *,
                 train: bool = False) -> Array:
        """
        Call Autoformer

        Parameters
        ----------
        seq : ArrayLike
            Inputs Signal. [B, I, d]
        cat : ArrayLike, optional
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

        B = inputs.shape[0]

        inputs = self.encode(seq, cat, with_dropout=train)
        assert inputs.shape == (B, self.I, self.d), "BUG"

        pred = self.decode(inputs, seq, cat, with_dropout=train)
        assert predh.shape == (B, self.O, self.d), "BUG"
        return pred

    @staticmethod
    def split_key(key: KeyArray, *,
                  train: bool = False) -> Tuple[KeyArray, Dict[str, KeyArray]]:
        """
        Split PRNG Key for Autoformer

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
            Keys can be consumed by Autoformer.
        """
        if train:
            key, key_a, key_d = jax.random.split(key, 2)
            return key, {"dropout": key_d}

        return key, dict()

    def log_model(self) -> None:
        """
        Log Informer Spec
        """
        logger.info("Autoformer(d=%d, I=%d, O=%d, Vs=%s, alpha=%f"
                    "nE=%d, nD=%d, nH=%d, dff=%d,"
                    " kMA=%d, eps=%.2e, Pdrop=%f)",
                    self.d, self.I, self.O, self.Vs, self.alpha,
                    self.nE, self.nD, self.nH, self.dff,
                    self.kMA, self.eps, self.Pdrop)
