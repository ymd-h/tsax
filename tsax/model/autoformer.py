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
import functools
import math
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
import wblog

from tsax.typing import Array, KeyArray
from tsax.core import (
    Model,
    ConvSeq,
    Embedding,
    FeedForward,
    MultiHeadAttention,
)
from tsax.core.encoding import EMBEDDING_ALPHA

__all__ = [
    "Autoformer",
    "EncodeStack",
    "DecoderStack",
    "EncoderLayer",
    "DecoderLayer",
    "AutoCorrelationAttention",
    "SeasonalLayerNorm",
    "SeriesDecomp",
]


logger = wblog.getLogger()


_UNUSED: int = -1

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
    def __call__(self, x: Array) -> Tuple[Array, Array]:
        """
        Call Series Decomposition

        Parameters
        ----------
        x : Array
            Input Series. [B, L, d]

        Returns
        -------
        season : Array
            Seasonal Component. [B, L, d]
        trend : Array
            Trend Component. [B, L, d]
        """
        B, L, d = x.shape
        left = (self.kMA -1) // 2
        right = (self.kMA -1) - left

        xpad = jnp.pad(x, ((0,0), (left, right), (0, 0)), mode="edge")

        trend = jax.vmap(
            lambda idx: (jax.lax.dynamic_slice(xpad, (0, idx, 0), (B, self.kMA, d))
                         .mean(axis=1)),
            out_axes=1
        )(jnp.arange(L))
        assert trend.shape == (B, L, d), f"BUG: {trend.shape} vs ({B}, {L}, {d})"

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
    def __call__(self, x: Array) -> Array:
        """
        Call Seasonal Layer Normalization

        Parameters
        ----------
        x : Array
            Input Sequence. [B, L, d]

        Returns
        -------
        x : Array
            Output Sequcence. [B, L, d]
        """
        x = nn.LayerNorm(epsilon=self.eps)(x)
        x = x.at[:].add(-jnp.mean(x, axis=1, keepdims=True))

        return  x


class AutoCorrelationAttention(nn.Module):
    """
    AutoCorrelation Attention

    Attributes
    ----------
    dk : int
        Key Dimension
    dv : int
        Value Dimension
    c : int
        Coefficient for Selecting Top K Correlation.
        floor(c * logL) Correlation is used.
        ``1 <= c <= 3``
    """
    dk: int
    dv: int
    c: int

    @nn.compact
    def __call__(self,
                 Q: Array,
                 K: Array,
                 V: Array) -> Array:
        """
        Call AutoCorrelation Attention Layer

        Paremeters
        ----------
        Q : Array
            Query. [B, L, dm]
        K : Array
            Key. [B, S, dm]
        V : Array
            Value. [B, S, dm]

        Returns
        -------
        A : Array
            Auto-Correlation as Attention. [B, L, d]
        """
        assert Q.shape[0] == K.shape[0] == V.shape[0], "BUG"
        assert K.shape[1] == V.shape[1], "BUG"
        assert Q.shape[2] == K.shape[2] == V.shape[2] == self.dk == self.dv, "BUG"

        B, L, _ = Q.shape

        # Resize by truncation or 0 filling.
        # K, V: [B, S, dm] -> [B, L, dm]
        K = jnp.zeros((B, L, self.dk), dtype=K.dtype).at[:,:K.shape[1],:].set(K)
        V = jnp.zeros((B, L, self.dv), dtype=V.dtype).at[:,:V.shape[1],:].set(V)
        assert K.shape == V.shape == (B, L, self.dk), f"BUG: {K.shape} vs {V.shape} vs {(B, L, self.dk)}"

        # Q_freq, K_freq: [B, L, dk]
        Q_freq = jnp.fft.rfft(Q, axis=1)
        K_freq = jnp.fft.rfft(K, axis=1)
        shape = (B, L//2+1, self.dk)
        msg = f"BUG: Q_freq: {Q_freq.shape} vs K_freq: {K_freq.shape} vs Expected: {shape}"
        assert Q_freq.shape == K_freq.shape == shape, msg

        # Rxx: [B, L, dk]
        Rxx = jnp.fft.irfft(Q_freq * jnp.conjugate(K_freq), n=L, axis=1)
        shape = (B, L, self.dk)
        assert Rxx.shape == shape, f"BUG: {Rxx.shape} vs {shape}"

        # Time Delay Aggregation
        # ----------------------

        # Note: Use ``math`` for static calculation
        k = min(int(math.floor(self.c * math.log(L))), L)

        Wk, Ik = jax.lax.top_k(jnp.moveaxis(Rxx, 1, -1), k)
        assert Wk.shape == Ik.shape == (B, self.dk, k), "BUG"

        Wk = nn.activation.softmax(Wk, axis=-1)

        @functools.partial(jax.vmap, in_axes=(0, 0, 1), out_axes=1)
        def _per_d(_w, _i, _v):
            assert _v.shape == (L,), "BUG"
            d_ret =  _w * jnp.roll(_v, -_i)
            assert d_ret.shape == _v.shape, "BUG"
            return d_ret

        @jax.vmap
        def _per_B(_wk, _ik, _V):
            assert _wk.shape == _ik.shape == (self.dk,), "BUG"
            assert _V.shape == (L, self.dv), "BUG"
            B_ret = _per_d(_wk, _ik, _V)
            assert B_ret.shape == (L, self.dv), "BUG"
            return B_ret

        @functools.partial(jax.vmap, in_axes=-1)
        def _per_k(wk, ik):
            assert wk.shape == ik.shape == (B, self.dk), "BUG"
            k_ret = _per_B(wk, ik, V)
            assert k_ret.shape == (B, L, self.dv), "BUG"
            return k_ret

        # A: [B, L, dv]
        A = jnp.sum(_per_k(Wk, Ik), axis=0)
        assert A.shape == (B, L, self.dv), "BUG"

        return A


class EncoderLayer(nn.Module):
    """
    Encoder Layer

    Attributes
    ----------
    c : int
        Coefficient for Selecting Top K Correlation.
        floor(c * logL) Correlation is used.
        ``1 <= c <= 3``
    dm : int
        Model Dimension
    nH : int
        Number of Multi Head
    kMA : int
        Window Size of Moving Average
    Pdrop : float
        Dropout Rate
    """
    c: int
    dm: int
    nH: int = NH
    dff: int = DFF
    kMA: int = K_MOVING_AVG
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
            Whether dropout or not.

        Returns
        -------
        inputs : Array
            Encoded Inputs. [B, L, dm]
        """
        shape = inputs.shape

        mha = MultiHeadAttention(
            attention=functools.partial(AutoCorrelationAttention, c=self.c),
            nH=self.nH,
            dm=self.dm,
            Pdrop=self.Pdrop
        )
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
    c : int
        Coefficient for Selecting Top K Correlation.
        floor(c * logL) Correlation is used.
        ``1 <= c <= 3``
    dm : int
        Model Dimension
    nH : int
        Number of Multi Head
    dff : int
        Number of Hidden Units at Feed Forward
    kMA : int
        Window Size of Moving Average
    Pdrop : float
        Dropout Rate
    """
    c: int
    dm: int
    nH: int = NH
    dff: int = DFF
    kMA: int = K_MOVING_AVG
    Pdrop: float = PDROP

    @nn.compact
    def __call__(self,
                 inputs: Array,
                 seasonal_outputs: Array,
                 trend_outputs: Array, *,
                 with_dropout: bool = False) -> Tuple[Array, Array]:
        """
        Call Decloder Layer

        Parameters
        ----------
        inputs : Array
            Encoded Inputs. [B, S, dm]
        seasonal_outputs : Array
            Seasonal Outputs. [B, L, dm]
        trend_outputs : Array
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

        s_mha = MultiHeadAttention(
            attention=functools.partial(AutoCorrelationAttention, c=self.c),
            dm=self.dm,
            nH=self.nH,
            Pdrop=self.Pdrop
        )
        c_mha = MultiHeadAttention(
            attention=functools.partial(AutoCorrelationAttention, c=self.c),
            dm=self.dm,
            nH=self.nH,
            Pdrop=self.Pdrop
        )
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
            seasonal_outputs + ff(seasonal_outputs, with_dropout=with_dropout)
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
    c : int
        Coefficient for Selecting Top K Correlation.
        floor(c * logL) Correlation is used.
        ``1 <= c <= 3``
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
    c : int
    dm: int
    nE: int = NE
    nH: int = NH
    dff: int = DFF
    kMA: int = K_MOVING_AVG
    eps: float = EPS
    Pdrop: float = PDROP

    @nn.compact
    def __call__(self,
                 inputs: Array,
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
        assert inputs.shape[2] == self.dm, "BUG"

        shape = inputs.shape

        E = EncoderLayer(c=self.c,
                         dm=self.dm,
                         nH=self.nH,
                         dff=self.dff,
                         kMA=self.kMA,
                         Pdrop=self.Pdrop)

        def f(enc: EncoderLayer, inp: Array, _: None) -> Tuple[Array, None]:
            inp = enc(inp, with_dropout=with_dropout)
            return inp, None

        inputs, _ = nn.scan(
            f,
            variable_axes={"params": 0},
            variable_broadcast=False,
            variable_carry=False,
            split_rngs={"params": True,
                        "dropout": True,
                        "attention": True},
            length=self.nE
        )(E, inputs, None)

        inputs = inputs.at[:].set(SeasonalLayerNorm(eps=self.eps)(inputs))

        assert inputs.shape == shape, "BUG"
        return inputs


class DecoderStack(nn.Module):
    """
    Decoder Stack

    Attributes
    ----------
    c : int
        Coefficient for Selecting Top K Correlation.
        floor(c * logL) Correlation is used.
        ``1 <= c <= 3``
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
    c: int
    dm: int
    nD: int = ND
    nH: int = NH
    dff: int = DFF
    kMA: int = K_MOVING_AVG
    eps: float = EPS
    Pdrop: float = PDROP

    @nn.compact
    def __call__(self,
                 inputs: Array,
                 seasonal_outputs: Array,
                 trend_outputs: Array, *,
                 with_dropout: bool = False) -> Tuple[Array, Array]:
        """
        Call Decoder Stack

        Parameters
        ----------
        inputs : Array
            Encoded Inputs. [B, S, dm]
        seasonal_outputs : Array
            Seasonal Outputs. [B, L, dm]
        trend_outputs : Array
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

        D = DecoderLayer(c=self.c,
                         dm=self.dm,
                         nH=self.nH,
                         dff=self.dff,
                         kMA=self.kMA,
                         Pdrop=self.Pdrop)

        def f(dec: DecoderLayer,
              st: Tuple[Array, Array],
              _: None) -> Tuple[Tuple[Array, Array], None]:
            s, t = st
            s, t = dec(inputs, s, t, with_dropout=with_dropout)
            return (s, t), None

        (seasonal_outputs, trend_outputs), _ = nn.scan(
            f,
            variable_axes={"params": 0},
            variable_broadcast=False,
            variable_carry=False,
            split_rngs={"params": True,
                        "dropout": True,
                        "attention": True},
            length=self.nD
        )(D, (seasonal_outputs, trend_outputs), None)

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
    c : int
        Coefficient for Selecting Top K Correlation.
        floor(c * logL) Correlation is used.
        ``1 <= c <= 3``
    """
    c: int
    d: int
    I: int
    O: int
    dm: int = _UNUSED
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
        self.encoder = EncoderStack(c=self.c,
                                    dm=self.d,
                                    nE=self.nE,
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

        self.decoder = DecoderStack(c=self.c,
                                    dm=self.d,
                                    nD=self.nD,
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
        self.decomp = SeriesDecomp(kMA=self.kMA)
        self.ff = nn.Dense(features=self.d)

    def encode(self,
               seq: Array,
               cat: Optional[Array] = None, *,
               with_dropout: bool = False) -> Array:
        """
        Encode with Autoformer

        Parameters
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
            Encoded Inputs. [B, I, d]
        """
        assert (cat is None) or seq.shape[:2] == cat.shape[:2], "BUG"
        assert seq.shape[1] == self.I, "BUG"

        B = seq.shape[0]

        inputs: Array = self.encoder_embed(seq, cat, with_dropout=with_dropout)
        assert inputs.shape == (B, self.I, self.d), "BUG"

        inputs = self.encoder(inputs, with_dropout=with_dropout)
        assert inputs.shape == (B, self.I, self.d), "BUG"

        return inputs

    def decode(self,
               inputs: Array,
               seq: Array,
               cat: Optional[Array] = None, *,
               with_dropout: bool = False) -> Array:
        """
        Decode with Autoformer

        Parameters
        ----------
        inputs : Array
            Encoded Inputs. [B, L, d]
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
        assert inputs.shape[2] == seq.shape[2], "BUG"
        assert (cat is None) or (seq.shape[:2] == cat.shape[:2]), "BUG"
        assert (cat is None) or (cat.shape[2] == len(self.Vs)), "BUG"

        B: int = inputs.shape[0]
        S: int = self.I // 2
        L: int = S + self.O


        s, t = self.decomp(seq.at[:,-S:,:].get())

        s_outputs = jnp.zeros((B, L, self.d), dtype=seq.dtype).at[:,:S,:].set(s)
        t_outputs = (jnp.zeros((B, L, self.d), dtype=seq.dtype)
                     .at[:,:S,:].set(t)
                     .at[:,S:,:].set(jnp.mean(seq, axis=1, keepdims=True)))

        # Only seasonal part is embedded.
        cat_pad = cat.at[:,-S:,:].get() if cat is not None else None
        s_outputs = self.decoder_embed(s_outputs, cat_pad, with_dropout=with_dropout)
        assert s_outputs.shape == (B, L, self.d)

        s_outputs, t_outputs = self.decoder(inputs, s_outputs, t_outputs,
                                            with_dropout=with_dropout)
        assert s_outputs.shape == t_outputs.shape == (B, L, self.d), "BUG"

        pred = s_outputs.at[:,-self.O:,:].get() + t_outputs.at[:,-self.O:,:].get()
        return pred

    def __call__(self,
                 seq: Array,
                 cat: Optional[Array] = None, *,
                 train: bool = False) -> Array:
        """
        Call Autoformer

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

        B = seq.shape[0]

        inputs = self.encode(seq, cat, with_dropout=train)
        assert inputs.shape == (B, self.I, self.d), "BUG"

        pred = self.decode(inputs, seq, cat, with_dropout=train)
        assert pred.shape == (B, self.O, self.d), "BUG"
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
            key, key_d = jax.random.split(key, 2)
            return key, {"dropout": key_d}

        return key, dict()

    def log_model(self) -> None:
        """
        Log Informer Spec
        """
        logger.info("Autoformer(d=%d, I=%d, O=%d, Vs=%s, alpha=%f,"
                    " nE=%d, nD=%d, nH=%d, dff=%d,"
                    " kMA=%d, eps=%.2e, Pdrop=%f)",
                    self.d, self.I, self.O, self.Vs, self.alpha,
                    self.nE, self.nD, self.nH, self.dff,
                    self.kMA, self.eps, self.Pdrop)
