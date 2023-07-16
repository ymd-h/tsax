from __future__ import annotations
from typing import Callable, Literal, Union

import jax
import jax.numpy as jnp
import flax.linen as nn

from tsax.typing import Array, ActivationFn
from tsax.typed_jax import relu, gelu

__all__ = [
    "ConvSeq",
    "FeedForward",
]


KERNEL_SIZE: int = 3
"""
Default Kernel Size for Convolution
"""

FF_KERNEL_SIZE: int = 1
"""
Default Kernel Size for Positional Feed Forward Layer
"""


class ConvSeq(nn.Module):
    """
    Convolution on Sequence

    Parameters
    ----------
    dm : int
        Model Dimension
    kernel : int
        Kernel Size for Convolution
    bias : bool
        Whether use bias or not
    """
    dm: int
    kernel: int = KERNEL_SIZE
    bias: bool = True

    @nn.compact
    def __call__(self, seq: Array) -> Array:
        """
        Convolute Sequnece

        Parameters
        ----------
        seq : Array
            Sequence. [B, L, d]

        Returns
        -------
        seq_conv : Array
            Convoluted Sequence. [B, L, dm]
        """
        seq = jnp.asarray(seq)

        conv = nn.Dense(features=self.dm, use_bias=self.bias)

        left: int = self.kernel // 2
        right: int = self.kernel - left

        seq_pad = jnp.pad(seq, ((0, 0), (left, right), (0, 0)), mode="wrap")

        idx = jnp.arange(seq.shape[1])
        seq_conv: Array = jax.vmap(
            lambda s: jax.vmap(
                lambda i: conv(
                    jnp.reshape(
                        jax.lax.dynamic_slice(s, (i, 0), (self.kernel, s.shape[1])),
                        (-1,)
                    )
                )
            )(idx)
        )(seq_pad)
        assert seq_conv.shape == (*seq.shape[:2], self.dm), f"BUG: {seq_conv.shape}"

        return seq_conv


class FeedForward(nn.Module):
    """
    Feed Forward Layer

    Parameters
    ----------
    dff : int
        Hidden Layer Units
    Pdrop : float
        Probability of Dropout
    activation : {"ReLU", "GELU"}
        Activation Function
    kernel : int
        Feed Forward Kernel Size
    bias : bool
        Whether use bias or not
    """
    dff: int
    Pdrop: float
    activation: Union[Literal["ReLU"], Literal["GELU"]] = "ReLU"
    kernel: int = FF_KERNEL_SIZE
    bias: bool = True

    @nn.compact
    def __call__(self, x: Array, *, train = False) -> Array:
        """
        Call Feed Foward Network

        Parameters
        ----------
        x : Array
            Inputs. [B, L, dm]
        train : bool, optional
            whether train or not

        Returns
        -------
        y : Array
            Outputs. [B, L, dm]
        """
        B, L, dm = x.shape

        activation: ActivationFn
        if self.activation == "ReLU":
            activation = relu
        elif self.activation == "GELU":
            activation = gelu
        else:
            raise ValueError("Unknown Activation")

        # h: [B, L, dff]
        h = activation(ConvSeq(dm=self.dff, kernel=self.kernel, bias=self.bias)(x))
        assert h.shape == (B, L, self.dff), "BUG"

        if train:
            h = h.at[:].set(nn.Dropout(self.Pdrop, deterministic=False)(h))

        y = ConvSeq(dm=dm, kernel=self.kernel, bias=self.bias)(h)
        assert y.shape == (B, L, dm), "BUG"

        if train:
            y = y.at[:].set(nn.Dropout(self.Pdrop, deterministic=False)(y))

        return y
