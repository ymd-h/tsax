"""
Transformer (:mod:`tsax.transformer`)
=====================================


Notes
-----
This module implements Transformer [1]_.


References
----------
.. [1] A. Vaswani et al., "Attention Is All You Need", NeurIPS 2017,
   https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html,
   https://arxiv.org/abs/1706.03762
"""
from __future__ import annotations
from typing import Optional

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from flax import linen as nn

from .core import ResidualLayerNorm

__all__ = [
    "Transformer",
    "Embedding",
    "EncoderStack",
    "DecoderStack",
    "EncoderLayer",
    "DecoderLayer",
    "MultiHeadAttention",
    "Attention",
    "FeedForward",
]

N: int = 6
"""
Default Number of Encoder/Decoder Stack
"""

DM: int = 512
"""
Default Model Dimension
"""

NH: int = 8
"""
Default Number of Multi Head Attention
"""

DFF: int = 2048
"""
Default Number of Units at Hidden Layer of Feed Forward
"""

EPS: float = 1e-12
"""
Default Value for Small Positive Value of Layer Normalization
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
    V : int
        Vocabulary Size
    L : int
        Tokenized Text Length
    dm : int
        Model Dimension
    Pdrop : float
        Dropout Rate
    """
    V: int
    L: int
    dm: int = DM
    Pdrop: float = PDROP

    def setup(self):
        self.embed = nn.Embed(self.V, self.dm)

        self.scale = jnp.sqrt(self.dm)

        half: int = self.dm // 2
        self.sin_dim: int = half + (self.dm % 2)
        self.cos_dim: int = half
        # `sin_dim` is equal to `cos_dim` or `cos_dim + 1`

        # freq: [sin_dim]
        self.freq = 1.0 / (10000 ** (2 * jnp.arange(self.sin_dim) / self.dm))

        self.drop = nn.Dropout(self.Pdrop, deterministic=False)

    def __call__(self, text: ArrayLike, with_dropout: bool = False) -> Array:
        """
        Call Embedding

        Parameters
        ----------
        text : ArrayLike
            Input Tokenized Text. [B, L]
        with_dropout : bool
            Whether to use dropout or not.

        Returns
        -------
        embedded : Array
            Embedded features with positional encoding. [B, L, dm]
        """
        # embedded: [B, L, dm]
        embedded = self.embed(text) * self.scale
        assert embedded.shape == (text.shape[0], self.L, self.dm), "BUG"

        # theta: [L, sin_dim]
        theta = jax.vmap(lambda pos: pos * self.freq)(jnp.arange(self.L))
        assert theta.shape == (self.L, self.sin_dim), "BUG"

        embedded = (embedded
                    .at[:,:,0::2].add(jnp.sin(theta))
                    .at[:,:,1::2].add(jnp.cos(theta.at[:,:self.cos_dim].get())))

        if with_dropout:
            embedded = embedded.at[:].set(self.drop(embedded))

        return embedded

    def attend(self, query: ArrayLike) -> Array:
        """
        De-embed Embedded Features

        Parameters
        ----------
        query : ArrayLike
            Query. [B, L, dm]

        Returns
        -------
        value : ArrayLike
            Attended. [B, L, V]
        """
        return self.embed.attend(query)


class FeedForward(nn.Module):
    """
    Position-wise Feed Forward Network

    Attributes
    ----------
    dff : int
        Number of Hidden Units at Feed Forward
    Pdrop : float
        Dropout Rate
    """
    dff: int = DFF
    Pdrop: float = PDROP

    @nn.compact
    def __call__(self, x: ArrayLike, *, with_dropout: bool = False) -> Array:
        """
        Call Position-wise Feed Forward Network

        Parameters
        ----------
        x : ArrayLike
            Inputs. [B, L, dm]
        with_dropout : bool, optional
            Whether dropout or not.

        Returns
        -------
        y : Array
            Outputs. [B, L, dm]
        """
        dm: int = x.shape[2]

        # h: [B, L, dff]
        h = nn.activation.relu(nn.Dense(self.dff)(x))

        # y: [B, L, dm]
        y = nn.Dense(dm)(h)

        if with_dropout:
            y = y.at[:].set(nn.Dropout(self.Pdrop, deterministic=False)(y))

        return y

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
                 V: ArrayLike,
                 mask: Optional[ArrayLike] = None) -> Array:
        """
        Call Attention Layer

        Parameters
        ----------
        Q : ArrayLike
            Query. [B, L, dm]
        K : ArrayLike
            Key. [B, L, dm]
        V : ArrayLike
            Value. [B, L, dm]
        mask : ArrayLike, optional
            Mask. [B, 1, L]/[B, L, L]

        Returns
        -------
        A : Array
            Attention. [B, L, dv]
        """
        assert Q.shape == K.shape == V.shape, "BUG"
        assert ((mask is None) or
                (mask.shape == (Q.shape[0], 1, Q.shape[1])) or
                (mask.shape == (Q.shape[0], Q.shape[1], Q.shape[1]))), "BUG"

        # Q, K: [B, L, dm] -> [B, L, dk]
        Q = nn.Dense(features=self.dk, use_bias=False, name="WQ")(Q)
        K = nn.Dense(features=self.dk, use_bias=False, name="WK")(K)

        # V: [B, L, dm] -> [B, L, dv]
        V = nn.Dense(features=self.dv, use_bias=False, name="WV")(V)

        # QK^T: [B, L, L]
        QK: Array = jnp.matmul(Q, jnp.transpose(K, (0, 2, 1)))
        assert QK.shape == (*Q.shape[:2], Q.shape[1]), "BUG"

        # Note: Python control flow is statically decided during `jit`-compiling.
        if mask is not None:
            QK = QK.at[:].set(jnp.where(mask==1, QK, -jnp.inf))

        QK = QK.at[:].divide(jnp.sqrt(self.dk))

        # A: [B, L, dv]
        A: Array = jnp.matmul(nn.activation.softmax(QK), V)
        assert A.shape == (*Q.shape[:2], self.dv)

        return A


class MultiHeadAttention(nn.Module):
    """
    Multi Head Attention Layer

    Attributes
    ----------
    nH : int
        Number of Multi Head
    dm : int
        Model Dimension
    Pdrop : float
        Dropout Rate
    """
    nH: int = NH
    dm: int = DM
    Pdrop: float = PDROP

    @nn.compact
    def __call__(self,
                 Q: ArrayLike,
                 K: ArrayLike,
                 V: ArrayLike,
                 mask: Optional[ArrayLike] = None, *,
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
        mask : ArrayLike, optional
            Batched Token Mask. [B, 1, L]/[B, L, L]
        with_dropout : bool, optional
            Whether dropout or not.

        Returns
        -------
        MHA : Array
            Multi Head Attention. [B, L, dm]
        """
        assert Q.shape == K.shape == V.shape, "BUG"
        assert ((mask is None) or
                (mask.shape == (Q.shape[0], 1, Q.shape[1])) or
                (mask.shape == (Q.shape[0], Q.shape[1], Q.shape[1]))), "BUG"

        # x: [B, L, dm (= dm/nH * nH)]
        d: int = self.dm // self.nH
        x = jnp.concatenate([Attention(dk=d, dv=d, name=f"head_{i}")(Q, K, V, mask)
                             for i in range(self.nH)],
                            axis=2)
        assert x.shape == (*Q.shape[:2], d * self.nH), "BUG"

        # MHA: [B, L, dm]
        MHA = nn.Dense(features=self.dm, use_bias=False, name="WO")(x)
        assert Q.shape == MHA.shape, "BUG"

        if with_dropout:
            MHA = MHA.at[:].set(nn.Dropout(self.Pdrop, deterministic=False)(MHA))

        return MHA


class EncoderLayer(nn.Module):
    """
    Encoder Layer

    Attributes
    ----------
    nH : int
        Number of Multi Head
    dm : int
        Model Dimension
    dff : int
        Number of Hidden Units at Feed Forward
    eps : float
        Small Positive Value for Layer Normalization
    Pdrop : float
        Dropout Rate
    """
    nH: int = NH
    dm: int = DM
    dff: int = DFF
    eps: float = EPS
    Pdrop: float = PDROP

    @nn.compact
    def __call__(self,
                 inputs: ArrayLike,
                 inputs_mask: ArrayLike, *,
                 with_dropout: bool = False) -> Array:
        """
        Call Encoder Layer

        Parameters
        ----------
        inputs : ArrayLike
            Inputs. [B, L, dm]
        inputs_mask : ArrayLike
            Padding Mask. [B, 1, L]
        with_dropout : bool, optional
            Whether dropout or not.

        Returns
        -------
        inputs : Array
            Encoded Inputs. [B, L, dm]
        """
        shape = inputs.shape

        mha = MultiHeadAttention(nH=self.nH, dm=self.dm, Pdrop=self.Pdrop)
        ff = FeedForward(dff=self.dff, Pdrop=self.Pdrop)

        # x: [B, L, m]
        inputs = ResidualLayerNorm(lambda i: mha(i, i, i, inputs_mask,
                                                 with_dropout=with_dropout),
                                   self.eps)(inputs)
        inputs = inputs.at[:].set(
            ResidualLayerNorm(lambda i: ff(i, with_dropout=with_dropout),
                              self.eps)(inputs)
        )

        assert inputs.shape == shape, "BUG"
        return inputs


class DecoderLayer(nn.Module):
    """
    Decoder Layer

    Attributes
    ----------
    nH : int
        Number of Multi Head
    dm : int
        Model Dimension
    dff : int
        Number of Hidden Units at Feed Forward
    eps : float
        Small Positive Value for Layer Normalization
    Pdrop : float
        Dropout Rate
    """
    nH: int = NH
    dm: int = DM
    dff: int = DFF
    eps: float = EPS
    Pdrop: float = PDROP

    @nn.compact
    def __call__(self,
                 inputs: ArrayLike,
                 inputs_mask: ArrayLike,
                 outputs: ArrayLike,
                 outputs_mask: ArrayLike, *,
                 with_dropout: bool = False) -> Array:
        """
        Call Decoder Layer

        Parameters
        ----------
        inputs : ArrayLike
            Encoded Inputs. [B, L, dm]
        inputs_mask : ArrayLike
            Padding Mask. [B, 1, L]
        outputs : ArrayLike
            Outputs. [B, L, dm]
        outputs_mask : ArrayLike
            Paddding & Subsequent Mask. [B, L, L]
        with_dropout : bool, optional
            Whether dropout or not.

        Returns
        -------
        outputs : Array
            Decoded Outputs. [B, L, dm]
        """
        assert inputs.shape == outputs.shape, "BUG"
        assert inputs.shape[:2] == mask.shape, "BUG"

        d: int = self.dm // self.nH
        mmha = MultiHeadAttention(nH=self.nH, dk=d, dv=d, Pdrop=self.Pdrop,
                                  name="MaskedMultiHeadAttention")
        mha  = MultiHeadAttention(nH=self.nH, dk=d, dv=d, Pdrop=self.Pdrop,
                                  name="MultiHeadAttention")
        ff = FeedForward(dff=self.dff, Pdrop=self.Pdrop)

        mmha_f = lambda o: mmha(o, o, o, outputs_mask, with_dropout=with_dropout)
        mha_f = lambda o: mha(o, inputs, inputs, inputs_mask, with_dropout=with_dropout)
        ff_f = lambda o: ff(o, with_dropout=with_dropout)

        outputs = ResidualLayerNorm(mmha_f, self.eps)(outputs)
        outputs = outputs.at[:].set(ResidualLayerNorm(mha_f, self.eps)(outputs))
        outputs = outputs.at[:].set(ResidualLayerNorm(ff_f, self.eps)(outputs))

        assert inputs.shape == outputs.shape, "BUG"
        return outputs


class EncoderStack(nn.Module):
    """
    Encoder Stack

    Attributes
    ----------
    N : int
        Number of Encoder Layers
    nH : int
        Number of Multi Head
    dm : int
        Model Dimension
    dff : int
        Number of Hidden Units at Feed Forward
    eps : float
        Small Positive Value for Layer Normalization
    Pdrop : float
        Dropout Rate
    """
    N: int = N
    nH: int = NH
    dm: int = DM
    dff: int = DFF
    eps: float = EPS
    Pdrop: float = PDROP

    @nn.compact
    def __call__(self,
                 inputs: ArrayLike,
                 mask: ArrayLike, *,
                 with_dropout: bool = False) -> Array:
        """
        Call Encoder Stack

        Parameters
        ----------
        inputs : ArrayLike
            Inputs. [B, L, dm]
        mask : ArrayLike
            Batched Token Mask. [B, L]
        with_dropout : bool, optional
            Whether dropout or not.

        Returns
        -------
        inputs : Array
            Encoded Inputs. [B, L, dm]
        """
        shape = inputs.shape

        # inputs_mask: [B, 1, L]
        # 'Inputs Mask' is 'Padding Mask' (will be broadcasted)
        inputs_mask = jnp.reshape(mask, (mask.shape[0], 1, mask.shape[1]))

        for i in range(self.N):
            inputs = EncoderLayer(nH=self.nH,
                                  dm=self.dm,
                                  dff=self.dff,
                                  eps=self.eps,
                                  Pdrop=self.Pdrop,
                                  name=f"EncoderLayer_{i}")(inputs,
                                                            inputs_mask,
                                                            with_dropout=with_dropout)

        assert inputs.shape == shape, "BUG"
        return inputs


class DecoderStack(nn.Module):
    """
    Decoder Stack

    Attributes
    ----------
    N : int
        Number of Decoder Layers
    nH : int
        Number of Multi Head
    dm : int
        Model Dimension
    dff : int
        Number of Hidden Units at Feed Forward
    eps : float
        Small Positive Value for Layer Normalization
    Pdrop : float
        Dropout Rate
    """
    N: int = N
    nH: int = NH
    dm: int = DM
    dff:int = DFF
    eps: float = EPS
    Pdrop: float = PDROP

    @nn.compact
    def __call__(self,
                 inputs: ArrayLike,
                 outputs: ArrayLike,
                 mask: ArrayLike, *,
                 with_dropout: bool = False) -> Array:
        """
        Call Decoder Stack

        Parameters
        ----------
        inputs : ArrrayLike
            Encoded Inputs. [B, L, dm]
        outputs : ArrayLike
            Outputs. [B, L, dm]
        mask : ArrayLike
            Batched Token Mask. [B, L]
        with_dropout : bool, optional
            Whether dropout or not.

        Returns
        -------
        outputs : Arrray
            Decoded Outputs. [B, L, dm]
        """
        assert inputs.shape == outputs.shape, "BUG"
        assert inputs.shape[:2] == mask.shape, "BUG"


        # inputs_mask: [B, 1, L]
        # 'Inputs Mask' is 'Padding Mask' (will be broadcasted)
        inputs_mask = jnp.reshape(mask, (mask.shape[0], 1, mask.shape[1]))

        # outputs_mask: [B, L, L]
        # 'Outputs Mask' is 'Padding Mask' & 'Subsequent Mask'
        outputs_mask = (inputs_mask *
                        jnp.tril(jnp.ones((mask.shape[1], mask.shape[1]), dtype=int)))


        for i in range(self.N):
            outputs = DecoderLayer(nH=self.nH,
                                   dm=self.dm,
                                   dff=self.dff,
                                   eps=self.eps,
                                   Pdrop=self.Pdrop,
                                   name=f"DecoderLayer_{i}")(inputs,
                                                             inputs_mask,
                                                             outputs,
                                                             outputs_mask,
                                                             with_dropout=with_dropout)

        assert inputs.shape == outputs.shape, "BUG"
        return outputs


class Transformer(nn.Module):
    """
    Transformer

    Attributes
    ----------
    V : int
        Vocabulary Size
    L : int
        Input Tokenized Text Length
    N : int
        Number of Layers at Encoder Stack & Decoder Stack
    dm : int
        Model Dimension
    nH : int
        Number of Multi Head
    dff : int
        Number of Hidden Units at Feed Forward
    eps : float
        Small Positive Value for Layer Normalization
    Pdrop : float
        Dropout Rate
    """
    V: int
    L: int
    N: int = N
    dm: int = DM
    nH: int = NH
    dff: int = DFF
    eps: float = EPS
    Pdrop: float = PDROP

    def setup(self):
        self.embed = Embedding(V=self.V, dm=self.dm, L=self.L, Pdrop=self.Pdrop)
        self.encoder = EncoderStack(N=self.N,
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

    def encode(self, inputs: ArrayLike, *, with_dropout: bool = False) -> Array:
        """
        Encode with Transformer

        Parameters
        ----------
        inputs : ArrayLike
            Batched Tokenized Input Text. [B, L]
        with_dropout : bool, optional
            Whether to use dropout or not.

        Returns
        -------
        inputs : Array
            Encoded Inputs
        """
        inputs = self.embed(inputs, with_dropout=with_dropout)

        # inputs: [B, L, dm]
        inputs = self.encoder(inputs, with_dropout=with_dropout)
        assert inputs.shape == (x.shape[0], self.L, self.dm), "BUG"

        return inputs

    def decode(self,
               inputs: ArrayLike,
               outputs: ArrayLike,
               mask: ArrayLike, *,
               with_dropout: bool = False,
               only_next: bool = False) -> Array:
        """
        Decode with Transformer

        Parameters
        ----------
        inputs : ArrayLike
            Batched Encoded Input. [B, L]
        outputs : ArrayLike
            Batched Tokenized Output Text. [B, L]
        mask : ArrayLike
            Batched Token Mask. [B, L]
        with_dropout : bool, optional
            Whether dropout or not.
        only_next : bool, optional
            Whether return only the next token or not.

        Returns
        -------
        p : Array
            Batched Token Probability. [B, L, V] / [B, V]
        """
        outputs = self.embed(outputs, with_dropout=with_dropout)

        # outputs: [B, L, dm]
        outputs = self.decoder(inputs, outputs, mask, with_dropout=with_dropout)
        assert outputs.shape == (x.shape[0], self.L, self.dm), "BUG"

        if only_next:
            # outputs: [B, dm]
            outputs = jnp.take(outputs, jnp.argmin(mask, axis=1), axis=1)
            assert outputs.shape == (x.shape[0], self.dm), "BUG"

        # p: [B, L, V] / [B, V]
        p = self.embed.attend(outputs)
        assert p.shape == (*outputs.shape[:-1], self.V), "BUG"

        p = nn.activation.softmax(p)
        return p

    def __call__(self,
                 inputs: ArrayLike,
                 outputs: ArrayLike,
                 mask: ArrayLike, *,
                 with_dropout: bool = False,
                 only_next: bool = False) -> Array:
        """
        Call Transformer

        Parameters
        ----------
        inputs : ArrayLike
            Batched Tokenized Input Text. [B, L]
        outputs : ArrayLike
            Batched Tokenized Output Text. [B, L]
        mask : ArrayLike
            Batched Token Mask. [B, L]
        with_dropout : bool, optional
            Whether dropout or not.
        only_next : bool, optional
            Whether return only the next token or not.

        Returns
        -------
        y : Array
            Batched Token Probability. [B, L, V] / [B, V]

        Notes
        -----
        This method calls `encode()` then `decode()` internally.
        """
        assert inputs.shape == outputs.shape, "BUG"
        assert inputs.shape[1] == self.L, "BUG"
        assert isinstance(with_dropout, bool), "BUG"

        inputs = self.encode(inputs, mask, with_dropout)

        return self.decode(inputs, outputs, mask,
                           with_dropout=with_dropout,
                           only_next=only_next)
