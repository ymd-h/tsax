import unittest

import jax
import jax.numpy as jnp

from tsax.transformer import (
    Transformer,
    Embedding,
    EncoderStack,
    EncoderLayer,
    DecoderStack,
    DecoderLayer,
    MultiHeadAttention,
    Attention,
    FeedForward,
)

class TestEmbedding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.B = 4
        cls.L = 12
        cls.V = 5
        cls.dm = 3

        key, key2 = jax.random.split(jax.random.PRNGKey(0), 2)

        cls.text = jax.random.randint(key, (cls.B, cls.L), 0, cls.V)
        cls.e = Embedding(V=cls.V, L=cls.L, dm=cls.dm)

        cls.params = cls.e.init(key2, cls.text)

    def test_without_jit(self):
        emb = self.e.apply(self.params, self.text)
        self.assertEqual(emb.shape, (self.B, self.L, self.dm))

        v = self.e.apply(self.params, emb, method=self.e.attend)
        self.assertEqual(v.shape, (self.B, self.L, self.V))

    def test_with_jit(self):
        emb = jax.jit(self.e.apply)(self.params, self.text)
        self.assertEqual(emb.shape, (self.B, self.L, self.dm))

        v = jax.jit(self.e.apply, static_argnames="method")(self.params,
                                                            emb,
                                                            method=self.e.attend)
        self.assertEqual(v.shape, (self.B, self.L, self.V))


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

        y = self.ff.apply(self.params, self.x, with_dropout=True, rngs={"dropout": key})
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
