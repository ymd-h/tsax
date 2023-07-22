import unittest

import jax
import jax.numpy as jnp

from tsax.core import ConvSeq, FeedForward
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


class TestFeedForward(TestCase):
    def test_ff(self):
        B, L, dm = 2, 12, 5
        dff = 32
        Pdrop = 0.8

        key = jax.random.PRNGKey(0)

        key, key_use = jax.random.split(key, 2)
        x = jax.random.normal(key_use, (B, L, dm))

        FF = FeedForward(dff=dff, Pdrop=Pdrop)

        key_p, key_d = jax.random.split(key, 2)

        ff, _ = FF.init_with_output(key_p, x)
        self.assertEqual(ff.shape, x.shape)

        ff_jit, _ = jax.jit(FF.init_with_output)(key_p, x)
        self.assertEqual(ff_jit.shape, x.shape)

        self.assertAllclose(ff, ff_jit)

        ff_drop, _ = FF.init_with_output({"params": key_p, "dropout": key_d},
                                         x, train=True)
        self.assertEqual(ff_drop.shape, x.shape)
        self.assertNotAllclose(ff, ff_drop)

        ff_drop_jit, _ = jax.jit(
            FF.init_with_output,
            static_argnames=["train"],
        )({"params": key_p, "dropout": key_d}, x, train=True)
        self.assertEqual(ff_drop_jit.shape, x.shape)
        self.assertNotAllclose(ff_jit, ff_drop_jit)

        self.assertAllclose(ff_drop, ff_drop_jit)


if __name__ == "__main__":
    unittest.main()
