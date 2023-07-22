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
from typing import cast, Optional

import jax
import jax.numpy as jnp
from flax import linen as nn

from tsax.typing import Array
from tsax.typed_jax import Dense
from tsax.core import ResidualLayerNorm, PositionalEncoding

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
    lazy_PE : bool
        Lazy Positional Encoding
    """
    V: int
    L: int
    dm: int = DM
    Pdrop: float = PDROP
    lazy_PE: bool = False

    def setup(self):
        self.embed = nn.Embed(self.V, self.dm)

        self.scale = jnp.sqrt(self.dm)

        self.pe = PositionalEncoding(dm=self.dm, L=self.L,
                                     Lfreq=10000, lazy=self.lazy_PE)

        self.drop = nn.Dropout(self.Pdrop, deterministic=False)

    def __call__(self, text: Array, *, train: bool = False) -> Array:
        """
        Call Embedding

        Parameters
        ----------
        text : Array
            Input Tokenized Text. [B, L]
        train : bool
            Whether to use dropout or not.

        Returns
        -------
        embedded : Array
            Embedded features with positional encoding. [B, L, dm]
        """
        # embedded: [B, L, dm]
        embedded: Array = self.embed(text) * self.scale
        assert embedded.shape == (text.shape[0], self.L, self.dm), "BUG"

        embedded = embedded.at[:,:,:].add(self.pe(embedded))

        if train:
            embedded = embedded.at[:].set(self.drop(embedded))

        return embedded

    def attend(self, query: Array) -> Array:
        """
        De-embed Embedded Features

        Parameters
        ----------
        query : Array
            Query. [B, L, dm]

        Returns
        -------
        value : Array
            Attended. [B, L, V]
        """
        return cast(Array, self.embed.attend(query))


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
    def __call__(self, x: Array, *, train: bool = False) -> Array:
        """
        Call Position-wise Feed Forward Network

        Parameters
        ----------
        x : Array
            Inputs. [B, L, dm]
        train : bool, optional
            whether train or not.

        Returns
        -------
        y : Array
            Outputs. [B, L, dm]
        """
        dm: int = x.shape[2]

        # h: [B, L, dff]
        h = nn.activation.relu(nn.Dense(self.dff)(x))

        # y: [B, L, dm]
        y: Array = nn.Dense(dm)(h)

        if train:
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
                 Q: Array,
                 K: Array,
                 V: Array,
                 mask: Optional[Array] = None) -> Array:
        """
        Call Attention Layer

        Parameters
        ----------
        Q : Array
            Query. [B, L, dm]
        K : Array
            Key. [B, L, dm]
        V : Array
            Value. [B, L, dm]
        mask : Array, optional
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
        Q = Dense(features=self.dk, use_bias=False, name="WQ")(Q)
        K = Dense(features=self.dk, use_bias=False, name="WK")(K)

        # V: [B, L, dm] -> [B, L, dv]
        V = Dense(features=self.dv, use_bias=False, name="WV")(V)

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
                 Q: Array,
                 K: Array,
                 V: Array,
                 mask: Optional[Array] = None, *,
                 train: bool = False) -> Array:
        """
        Multi Head Attention

        Parameters
        ----------
        Q : Array
            Query. [B, L, dm]
        K : Array
            Key. [B, L, dm]
        V : Array
            Value. [B, L, dm]
        mask : Array, optional
            Batched Token Mask. [B, 1, L]/[B, L, L]
        train : bool, optional
            whether train or not.

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
        x = jnp.concatenate([
            Attention(dk=d, dv=d, name=f"head_{i}")( # type: ignore[call-arg]
                Q, K, V, mask
            )
            for i in range(self.nH)
        ], axis=2)
        assert x.shape == (*Q.shape[:2], d * self.nH), "BUG"

        # MHA: [B, L, dm]
        MHA: Array = Dense(features=self.dm, use_bias=False, name="WO")(x)
        assert Q.shape == MHA.shape, "BUG"

        if train:
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
                 inputs: Array,
                 inputs_mask: Array, *,
                 train: bool = False) -> Array:
        """
        Call Encoder Layer

        Parameters
        ----------
        inputs : Array
            Inputs. [B, L, dm]
        inputs_mask : Array
            Padding Mask. [B, 1, L]
        train : bool, optional
            whether train or not.

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
                                                 train=train),
                                   self.eps)(inputs)
        inputs = inputs.at[:].set(
            ResidualLayerNorm(lambda i: ff(i, train=train),
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
                 inputs: Array,
                 inputs_mask: Array,
                 outputs: Array,
                 outputs_mask: Array, *,
                 train: bool = False) -> Array:
        """
        Call Decoder Layer

        Parameters
        ----------
        inputs : Array
            Encoded Inputs. [B, L, dm]
        inputs_mask : Array
            Padding Mask. [B, 1, L]
        outputs : Array
            Outputs. [B, L, dm]
        outputs_mask : Array
            Paddding & Subsequent Mask. [B, L, L]
        train : bool, optional
            whether train or not.

        Returns
        -------
        outputs : Array
            Decoded Outputs. [B, L, dm]
        """
        assert inputs.shape == outputs.shape, "BUG"
        assert (inputs.shape[:2] ==
                (inputs_mask.shape[0], inputs_mask.shape[2]) ==
                (outputs_mask.shape[0], outputs_mask.shape[2])), "BUG"
        assert ((inputs_mask.shape[1] in [1, inputs.shape[1]]) and
                (outputs_mask.shape[1] in [1, outputs.shape[1]])), "BUG"

        mmha = MultiHeadAttention(nH=self.nH, # type: ignore[call-arg]
                                  dm=self.dm, Pdrop=self.Pdrop,
                                  name="MaskedMultiHeadAttention")
        mha  = MultiHeadAttention(nH=self.nH, # type: ignore[call-arg]
                                  dm=self.dm, Pdrop=self.Pdrop,
                                  name="MultiHeadAttention")
        ff = FeedForward(dff=self.dff, Pdrop=self.Pdrop)

        mmha_f = lambda o: mmha(o, o, o, outputs_mask, train=train)
        mha_f = lambda o: mha(o, inputs, inputs, inputs_mask, train=train)
        ff_f = lambda o: ff(o, train=train)

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
                 inputs: Array,
                 mask: Array, *,
                 train: bool = False) -> Array:
        """
        Call Encoder Stack

        Parameters
        ----------
        inputs : Array
            Inputs. [B, L, dm]
        mask : Array
            Batched Token Mask. [B, L]
        train : bool, optional
            whether train or not.

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
            inputs = EncoderLayer(nH=self.nH, # type: ignore
                                  dm=self.dm,
                                  dff=self.dff,
                                  eps=self.eps,
                                  Pdrop=self.Pdrop,
                                  name=f"EncoderLayer_{i}")(inputs,
                                                            inputs_mask,
                                                            train=train)

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
                 inputs: Array,
                 inputs_mask: Array,
                 outputs: Array,
                 outputs_mask: Array, *,
                 train: bool = False) -> Array:
        """
        Call Decoder Stack

        Parameters
        ----------
        inputs : ArrrayLike
            Encoded Inputs. [B, L, dm]
        inputs_mask : Array
            Batched Token Mask for Inputs. [B, L]
        outputs : Array
            Outputs. [B, L, dm]
        outputs_mask : Array
            Batched Token Mask for Outputs. [B, L]
        train : bool, optional
            whether train or not.

        Returns
        -------
        outputs : Arrray
            Decoded Outputs. [B, L, dm]
        """
        assert inputs.shape == outputs.shape, "BUG"
        assert inputs.shape[:2] == inputs_mask.shape == outputs_mask.shape, "BUG"

        B = inputs_mask.shape[0]
        L = inputs_mask.shape[1]

        # inputs_mask: [B, 1, L]
        # 'Inputs Mask' is 'Padding Mask' (will be broadcasted)
        inputs_mask = jnp.reshape(inputs_mask, (B, 1, L))

        # outputs_mask: [B, L, L]
        # 'Outputs Mask' is 'Padding Mask' & 'Subsequent Mask'
        outputs_mask = (jnp.reshape(outputs_mask, (B, 1, L)) *
                        jnp.tril(jnp.ones((L, L), dtype=int)))


        for i in range(self.N):
            outputs = DecoderLayer(nH=self.nH, # type: ignore
                                   dm=self.dm,
                                   dff=self.dff,
                                   eps=self.eps,
                                   Pdrop=self.Pdrop,
                                   name=f"DecoderLayer_{i}")(inputs,
                                                             inputs_mask,
                                                             outputs,
                                                             outputs_mask,
                                                             train=train)

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

    def encode(self,
               inputs: Array,
               inputs_mask, *,
               train: bool = False) -> Array:
        """
        Encode with Transformer

        Parameters
        ----------
        inputs : Array
            Batched Tokenized Input Text. [B, L]
        inputs_mask : Array
            Batched Token Mask for Inputs. [B, L]
        train : bool, optional
            Whether to use dropout or not.

        Returns
        -------
        inputs : Array
            Encoded Inputs
        """
        inputs = self.embed(inputs, train=train)

        # inputs: [B, L, dm]
        inputs = self.encoder(inputs, inputs_mask, train=train)
        assert inputs.shape == (inputs.shape[0], self.L, self.dm), "BUG"

        return inputs

    def decode(self,
               inputs: Array,
               inputs_mask: Array,
               outputs: Array,
               outputs_mask: Array, *,
               train: bool = False,
               only_next: bool = False) -> Array:
        """
        Decode with Transformer

        Parameters
        ----------
        inputs : Array
            Batched Encoded Input. [B, L]
        inputs_mask : Array
            Batched Token Mask for Inputs. [B, L]
        outputs : Array
            Batched Tokenized Output Text. [B, L]
        outputs_mask : Array
            Batched Token Mask for Outputs. [B, L]
        train : bool, optional
            whether train or not.
        only_next : bool, optional
            Whether return only the next token or not.

        Returns
        -------
        p : Array
            Batched Token Probability. [B, L, V] / [B, V]
        """
        B = inputs.shape[0]

        outputs = self.embed(outputs, train=train)

        # outputs: [B, L, dm]
        outputs = self.decoder(inputs, inputs_mask,
                               outputs, outputs_mask,
                               train=train)
        assert outputs.shape == (B, self.L, self.dm), "BUG"

        if only_next:
            f = jax.vmap(lambda o, idx: o.at[idx, :].get())

            # outputs: [B, dm]
            outputs =  f(outputs, jnp.argmin(outputs_mask, axis=1))
            assert outputs.shape == (B, self.dm), f"BUG: {outputs.shape}"

        # p: [B, L, V] / [B, V]
        p: Array = self.embed.attend(outputs)
        assert p.shape == (*outputs.shape[:-1], self.V), "BUG"

        p = nn.activation.softmax(p)
        return p

    def __call__(self,
                 inputs: Array,
                 inputs_mask: Array,
                 outputs: Array,
                 outputs_mask: Array, *,
                 train: bool = False,
                 only_next: bool = False) -> Array:
        """
        Call Transformer

        Parameters
        ----------
        inputs : Array
            Batched Tokenized Input Text. [B, L]
        inputs_mask : Array
            Batched Token Mask for Inputs. [B, L]
        outputs : Array
            Batched Tokenized Output Text. [B, L]
        outputs_mask : Array
            Batched Token Mask for Outputs. [B, L]
        train : bool, optional
            whether train or not.
        only_next : bool, optional
            Whether return only the next token or not.

        Returns
        -------
        p : Array
            Batched Token Probability. [B, L, V] / [B, V]

        Notes
        -----
        This method calls `encode()` then `decode()` internally.
        """
        assert inputs.shape == outputs.shape, "BUG"
        assert inputs.shape[1] == self.L, "BUG"
        assert isinstance(train, bool), "BUG"

        inputs = self.encode(inputs, inputs_mask, train=train)

        return self.decode(inputs, inputs_mask, outputs, outputs_mask,
                           train=train,
                           only_next=only_next)
