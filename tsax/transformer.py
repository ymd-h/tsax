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
    """
    V: int
    L: int
    dm: int = DM

    @nn.compact
    def __call__(self, text: ArrayLike) -> Array:
        """
        Embed

        Parameters
        ----------
        text : ArrayLike
            Input Tokenized Text. [B, L]

        Returns
        -------
        embedded : Array
            Embedded features with positional encoding. [B, L, dm]
        """
        # embedded: [B, L, dm]
        embedded = nn.Embed(self.V, self.dm)(text)
        assert embedded.shape == (text.shape[0], self.L, self.dm), "BUG"

        half: int = self.dm // 2
        sin_dim: int = half + (self.dm % 2)
        cos_dim: int = half
        # `sin_dim` is equal to `cos_dim` or `cos_dim + 1`

        # exponent: [sin_dim]
        exponent = 2 * jnp.arange(sin_dim) / self.dm

        # theta: [L, sin_dim]
        theta = jax.vmap(lambda pos: (pos / 10000) ** exponent)(jnp.arange(self.L))
        assert theta.shape == (self.L, sin_dim), "BUG"

        embedded = (embedded
                    .at[:,:,0::2].add(jnp.sin(theta))                       # Even
                    .at[:,:,1::2].add(jnp.cos(theta.at[:,:cos_dim].get()))) # Odd

        return embedded


class FeedForward(nn.Module):
    """
    Position-wise Feed Forward Network

    Attributes
    ----------
    dff : int
        Number of Hidden Units at Feed Forward
    """
    dff: int = DFF

    @nn.compact
    def __call__(self, x: ArrayLike) -> Array:
        """
        Call Position-wise Feed Forward Network

        Parameters
        ----------
        x : ArrayLike
            Inputs. [B, L, dm]

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
                 mask: Optional[ArrayLike]) -> Array:
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
            Mask. [B, L]

        Returns
        -------
        A : Array
            Attention. [B, dv]
        """
        assert Q.shape == K.shape == V.shape, "BUG"
        assert Q.shape[:2] == mask.shape, "BUG"
        
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
            # mask: [B, L] -> [B, L, L]
            mask = jnp.reshape(mask, mask.shape[0], 1, mask.shape[1])
            QK = QK.at[:].set(jnp.where(mask==1, QK, -jnp.inf))

        QK = QK.at[:].div(jnp.sqrt(self.dk))

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
    """
    nH: int = NH
    dm: int = DM

    @nn.compact
    def __call__(self,
                 Q: ArrayLike,
                 K: ArrayLike,
                 V: ArrayLike,
                 mask: Optional[ArrayLike] = None) -> Array:
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
            Batched Token Mask. [B, L]

        Returns
        -------
        MHA : Array
            Multi Head Attention. [B, L, dm]
        """
        assert Q.shape == K.shape == V.shape, "BUG"
        assert Q.shape[:2] == mask.shape, "BUG"

        # x: [B, L, dm (= dm/nH * nH)]
        d: int = self.dm // self.nH
        x = jnp.concatenate((Attention(dk=d, dv=d, name=f"head_{i}")(Q, K, V, mask)
                             for i in range(self.nH)),
                            axis=2)
        assert x.shape == (*Q.shape[:2], d * self.nH), "BUG"

        # MHA: [B, L, dm]
        MHA = nn.Dense(features=self.dm, use_bias=False, name="WO")(x)
        assert Q.shape == MHA.shape, "BUG"

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
    """
    nH: int = NH
    dm: int = DM
    dff: int = DFF
    eps: float = EPS

    @nn.compact
    def __call__(self, inputs: ArrayLike) -> Array:
        """
        Call Encoder Layer

        Parameters
        ----------
        inputs : ArrayLike
            Inputs. [B, L, dm]

        Returns
        -------
        inputs : Array
            Encoded Inputs. [B, L, dm]
        """
        shape = inputs.shape

        mha = MultiHeadAttention(nH=self.nH, dm=self.dm)
        ff = FeedForward(dff=self.dff)

        # x: [B, L, m]
        inputs = nn.LayerNorm(epsilon=self.eps)(inputs + mha(inputs, inputs, inputs))
        inputs = x.at[:,:,:].set(
            nn.LayerNorm(epsilon=self.eps)(inputs + ff(inputs))
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
    """
    nH: int = NH
    dm: int = DM
    dff: int = DFF
    eps: float = EPS

    @nn.compact
    def __call__(self,
                 inputs: ArrayLike,
                 outputs: ArrayLike,
                 mask: ArrayLike) -> Array:
        """
        Call Decoder Layer

        Parameters
        ----------
        inputs : ArrayLike
            Encoded Inputs. [B, L, dm]
        outputs : ArrayLike
            Outputs. [B, L, dm]
        mask : ArrayLike
            Batched Token Mask. [B, L]

        Returns
        -------
        outputs : Array
            Decoded Outputs. [B, L, dm]
        """
        assert inputs.shape == outputs.shape, "BUG"
        assert inputs.shape[:2] == mask.shape, "BUG"

        d: int = self.dm // self.nH
        mmha = MultiHeadAttention(nH=self.nH, dk=d, dv=d,
                                  name="MaskedMultiHeadAttention")
        mha = MultiHeadAttention(nH=self.nH, dk=d, dv=d, name="MultiHeadAttention")
        ff = FeedForward(dff=self.dff)

        outputs = nn.LayerNorm(epsilon=self.eps)(outputs + mmha(outputs,
                                                                outputs,
                                                                outputs,
                                                                mask))
        outputs = nn.LayerNorm(epsilon=self.eps)(outputs + mha(inputs,
                                                               inputs,
                                                               outputs))
        outputs = nn.LayerNorm(epsilon=self.eps)(outputs + ff(outputs))

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
    """
    N: int = N
    nH: int = NH
    dm: int = DM
    dff: int = DFF
    eps: float = EPS

    @nn.compact
    def __call__(self, inputs: ArrayLike) -> Array:
        """
        Call Encoder Stack

        Parameters
        ----------
        inputs : ArrayLike
            Inputs. [B, L, dm]

        Returns
        -------
        inputs : Array
            Encoded Inputs. [B, L, dm]
        """
        shape = inputs.shape

        for i in range(self.N):
            inputs = EncoderLayer(nH=self.nH,
                                  dm=self.dm,
                                  dff=self.dff,
                                  eps=self.eps,
                                  name=f"EncoderLayer_{i}")(inputs)

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
    """
    N: int = N
    nH: int = NH
    dm: int = DM
    dff:int = DFF
    eps: float = EPS

    @nn.compact
    def __call__(self,
                 inputs: ArrayLike,
                 outputs: ArrayLike,
                 mask: ArrayLike) -> Array:
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

        Returns
        -------
        outputs : Arrray
            Decoded Outputs. [B, L, dm]
        """
        assert inputs.shape == outputs.shape, "BUG"
        assert inputs.shape[:2] == mask.shape, "BUG"

        for i in range(self.N):
            outputs = DecoderLayer(nH=self.nH,
                                   dm=self.dm,
                                   dff=self.dff,
                                   eps=self.eps,
                                   name=f"DecoderLayer_{i}")(inputs, outputs, mask)

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
    """
    V: int
    L: int
    N: int = N
    dm: int = DM
    nH: int = NH
    dff: int = DFF
    eps: float = EPS


    @nn.compact
    def __call__(self, x: ArrayLike, mask: ArrayLike) -> Array:
        """
        Transformer

        Parameters
        ----------
        x : ArrayLike
            Batched Tokenized Text. [B, L]
        mask : ArrayLike
            Batched Token Mask. [B, L]

        Returns
        -------
        y : Array
            Batched Token Probability. [B, L, V]
        """
        assert x.shape[1] == self.L

        inputs, outputs = x, x

        embed = Embedding(V=self.V, dm=self.dm, L=self.L)

        # Share Embedding Weight Matrix
        inputs = embed(inputs)
        outputs = embed(outputs)

        # inputs: [B, L, dm]
        inputs = EncoderStack(N=self.N,
                              dm=self.dm,
                              nH=self.nH,
                              dff=self.dff,
                              eps=self.eps)(inputs)
        assert inputs.shape == (x.shape[0], self.L, self.dm), "BUG"

        # outputs: [B, L, dm]
        outputs = DecoderStack(N=self.N,
                               dm=self.dm,
                               nH=self.nH,
                               dff=self.dff,
                               eps=self.eps)(inputs, outputs, mask)
        assert outputs.shape == (x.shape[0], self.L, self.dm), "BUG"

        # y: [B, L, V]
        y = embed.attend(outputs)
        assert y.shape == (x.shape[0], self.L, self.V), "BUG"

        # y: [B, L, V]
        y = nn.activation.softmax(y)
        return y
