"""
Attention (:mod:`tsax.core.attention`)
"""
from __future__ import annotations
from functools import partial
from typing import Union, Type

import jax
import jax.numpy as jnp
import flax.linen as nn


from tsax.typing import Array
from tsax.typed_jax import Dense


__all__ = [
    "MultiHeadAttention",
]


class MultiHeadAttention(nn.Module):
    """
    Multi Head Attention Layer

    Parameters
    ----------
    attention: nn.Module
        Attention
    nH : int
        Number of Multi Head
    dm : int
        Model Dimension
    Pdrop : float
        Dropout Probability
    bias : bool
        Whether use bias or not
    """
    attention: Union[Type[nn.Module], partial[nn.Module]]
    dm: int
    nH: int
    Pdrop: float
    bias: bool = False

    @nn.compact
    def __call__(self,
                 Q: Array,
                 K: Array,
                 V: Array, *,
                 train: bool = False) -> Array:
        """
        Call Multi Head Attention

        Parameters
        ----------
        Q : ArrayLike
            Query. [B, Lq, dm]
        K : ArrayLike
            Key. [B, Lk, dm]
        V : ArrayLike
            Value. [B, Lk, dm]
        train : bool, optional
            whether train or not

        Returns
        -------
        MHA : Array
            Multi Head Attention. [B, L, dm]
        """
        assert K.shape == V.shape, "BUG"
        assert Q.shape[0] == K.shape[0], "BUG"
        assert Q.shape[2] == K.shape[2], "BUG"

        B, L, _ = Q.shape
        _, S, _ = K.shape

        # x: [B, L, dm (= dm/nH * nH)]
        d: int = max(self.dm // self.nH, 1)

        A = nn.vmap(self.attention,
                    in_axes=-1, out_axes=-1,
                    variable_axes={"params": 0},
                    split_rngs={"params": True,
                                "dropout": True,
                                "attention": True})

        Q = jnp.reshape(Dense(features=d*self.nH, use_bias=self.bias, name="WQ")(Q),
                        (B, L, d, self.nH))
        K = jnp.reshape(Dense(features=d*self.nH, use_bias=self.bias, name="WK")(K),
                        (B, S, d, self.nH))
        V = jnp.reshape(Dense(features=d*self.nH, use_bias=self.bias, name="WV")(V),
                        (B, S, d, self.nH))

        a = A(dk=d, dv=d)(Q, K, V)
        assert a.shape == (B, L, d, self.nH), "BUG"

        a = jnp.reshape(a, (B, L, d*self.nH))

        MHA: Array = Dense(features=self.dm, name="WO", use_bias=self.bias)(a)
        assert MHA.shape == (B, L, self.dm), "BUG"

        if train:
            MHA = MHA.at[:].set(nn.Dropout(self.Pdrop, deterministic=False)(MHA))

        return MHA
