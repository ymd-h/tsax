from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn


KERNEL_SIZE: int = 3
"""
Default Kernel Size for Convolution
"""

class ConvSeq(nn.Module):
    """
    Convolution on Sequence
    """
    dm: int
    kernel: int = KERNEL_SIZE

    @nn.compact
    def __call__(self, seq: ArrayLike) -> Array:
        """
        Convolute Sequnece

        Parameters
        ----------
        seq : ArrayLike
            Sequence. [B, L, d]

        Returns
        -------
        seq_conv : Array
            Convoluted Sequence. [B, L, dm]
        """
        conv = nn.Dense(features=self.dm)

        left: int = self.kernel // 2
        right: int = self.kernel - left

        seq_pad = jnp.pad(seq, ((0, 0), (left, right), (0, 0)), mode="wrap")

        idx = jnp.arange(seq.shape[1])
        seq_conv = jax.vmap(
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
