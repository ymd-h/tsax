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
    def test_without_jit(self):
        B = 4
        L = 12
        V = 5
        dm = 3

        key = jax.random.PRNGKey(0)
        text = jnp.zeros((B, L), dtype=int)

        e = Embedding(V=V, L=L, dm=dm)
        params = e.init(key, text)

        emb = e.apply(params, text)
        self.assertEqual(emb.shape, (B, L, dm))

        v = e.apply(params, emb, method=e.attend)
        self.assertEqual(v.shape, (B, L, V))

    def test_with_jit(self):
        B = 4
        L = 12
        V = 5
        dm = 3

        key = jax.random.PRNGKey(0)
        text = jnp.zeros((B, L), dtype=int)

        e = Embedding(V=V, L=L, dm=dm)
        params = e.init(key, text)

        emb = jax.jit(e.apply)(params, text)
        self.assertEqual(emb.shape, (B, L, dm))

        v = jax.jit(e.apply, static_argnames="method")(params, emb, method=e.attend)
        self.assertEqual(v.shape, (B, L, V))


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
