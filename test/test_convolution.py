import unittest

import jax
import jax.numpy as jnp

from tsax.core import ConvSeq
from tsax.testing import TestCase


class TestConvolution(TestCase):
    def test_conv(self):
        B, L, d, dm = 1, 3, 1, 3

        key = jax.random.PRNGKey(0)

        key, key_use = jax.random.split(key, 2)
        x = jax.random.normal(key_use, (B, L, d))

        conv = ConvSeq(dm=dm)

        key, key_use = jax.random.split(key, 2)
        x_conv, _ = conv.init_with_output(key_use, x)
        self.assertEqual(x_conv.shape, (B, L, dm))

        x_conv_jit, _ = jax.jit(conv.init_with_output)(key_use, x)
        self.assertEqual(x_conv_jit.shape, (B, L, dm))

        self.assertAllclose(x_conv, x_conv_jit)

    def test_multi_conv(self):
        B, L, d, dm = 1, 3, 5, 3

        key = jax.random.PRNGKey(0)

        key, key_use = jax.random.split(key, 2)
        x = jax.random.normal(key_use, (B, L, d))

        conv = ConvSeq(dm=dm)

        key, key_use = jax.random.split(key, 2)
        x_conv, _ = conv.init_with_output(key_use, x)
        self.assertEqual(x_conv.shape, (B, L, dm))

        x_conv_jit, _ = jax.jit(conv.init_with_output)(key_use, x)
        self.assertEqual(x_conv_jit.shape, (B, L, dm))

        self.assertAllclose(x_conv, x_conv_jit)



if __name__ == "__main__":
    unittest.main()
