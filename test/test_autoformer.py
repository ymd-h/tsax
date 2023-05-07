import unittest

import jax
import jax.numpy as jnp

from tsax.autoformer import (
    Autoformer,
    EncoderStack,
    DecoderStack,
    EncoderLayer,
    DecoderLayer,
    MultiHeadAttention,
    AutoCorrelationAttention,
    SeriesDecomp,
    FeedForward,
)


class TestFeedForward(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.B = 4
        cls.L = 12
        cls.dm = 3
        cls.dff = 24
        cls.Pdrop = 0.99

        key, key2, key3 = jax.random.split(jax.random.PRNGKey(0), 3)

        cls.x = jax.random.normal(key3, (cls.B, cls.L, cls.dm))

        cls.ff = FeedForward(dff=cls.dff, Pdrop=cls.Pdrop)
        cls.params = cls.ff.init(key, cls.x)
        cls.key = key2

    def test_without_jit(self):
        y = self.ff.apply(self.params, self.x)
        self.assertEqual(y.shape, self.x.shape)

        self.assertTrue(jnp.all(y == self.ff.apply(self.params, self.x)))

    def test_with_jit(self):
        f = jax.jit(self.ff.apply)
        y = f(self.params, self.x)
        self.assertEqual(y.shape, self.x.shape)

        self.assertTrue(jnp.all(y == f(self.params, self.x)))

    def test_dropout_without_jit(self):
        key, key2 = jax.random.split(self.key)

        y = self.ff.apply(self.params, self.x, with_dropout=True,
                          rngs={"dropout": key})
        self.assertEqual(y.shape, self.x.shape)

        self.assertFalse(jnp.all(y == self.ff.apply(self.params,
                                                    self.x,
                                                    with_dropout=True,
                                                    rngs={"dropout": key2})))

    def test_dropout_with_jit(self):
        key, key2 = jax.random.split(self.key)

        f = jax.jit(self.ff.apply, static_argnames="with_dropout")
        y = f(self.params, self.x, with_dropout=True, rngs={"dropout": key})
        self.assertEqual(y.shape, self.x.shape)

        self.assertFalse(jnp.all(y == f(self.params,
                                        self.x,
                                        with_dropout=True,
                                        rngs={"dropout": key2})))


if __name__ == "__main__":
    unittest.main()
