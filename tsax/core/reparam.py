"""
Reparam (:mod:`tsax.core.reparam`)
==================================
"""
from __future__ import annotations
from typing import Optional

import jax
import jax.numpy as jnp
import flax.linen as nn

from tsax.typing import Array

__all__ = [
    "SigmaReparamDense",
]



class SigmaReparamDense(nn.Module):
    """
    Dense with Sigma Reparam

    Notes
    -----
    To stabilize Transformer by keeping Attention's entropy,
    use Spectral Normalization [1]_.

    References
    ----------
    .. [1] S. Zhang et al., "Stabilizing Transformer Training
       by Preventing Attention Entropy Collapse", ICML 2023
       https://proceedings.mlr.press/v202/zhai23a.html,
       [arXiv:2303.06296](https://arxiv.org/abs/2303.06296)
    """
    features: int
    use_bias: bool = True
    name: Optional[str] = None
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    precision: PrecisionLike = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros_init()
    gamma_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.ones
    dot_general: DotGeneralT = jax.lax.dot_general

    @compact
    def __call__(self,
                 inputs: Array, *,
                 training: bool = False) -> Array:
        """Applies a linear transformation to the inputs along the last dimension.

        Parameters
        ----------
        inputs : Array
            The nd-array to be transformed.

        Returns
        -------
        Array
            The transformed input.
        """
        kernel = self.param('kernel',
                            self.kernel_init,
                            (jnp.shape(inputs)[-1], self.features),
                            self.param_dtype)
        if self.use_bias:
            bias = self.param('bias', self.bias_init, (self.features,),
                              self.param_dtype)
        else:
            bias = None

        u = self.variable("sigma_reparam", "u",
                          lambda s: jax.random.normal(self.make_rng("param"), (s,)),
                          kernel.shape[0])
        v = self.variable("sigma_reparam", "v",
                          lambda s: jax.random.normal(self.make_rng("param"), (s,)),
                          kernel.shape[1])

        uWv = self.variable("sigma_reparam", "uWv",
                            lambda _: u.value @ kernel @ v.value, None)

        if training:
            u.value = jax.lax.stop_gradient(kernel @ v.value)
            u.value.at[:].divide(jnp.linalg.norm(u.value, keepdims=True))

            v.value = jax.lax.stop_gradient(jnp.transpose(kernel) @ u.value)
            v.value.at[:].divide(jnp.linalg.norm(v.value, keepdims=True))

            uWv.value = u.value @ kernel @ v.value

        gamma = self.param("gamma", self.gamma_init, (1,), self.param_dtype)

        inputs, kernel, bias = promote_dtype(inputs,
                                             gamma / uWv.value * kernel,
                                             bias,
                                             dtype=self.dtype)

        y = self.dot_general(
            inputs,
            kernel,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )

        if bias is not None:
            y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))

        return y
