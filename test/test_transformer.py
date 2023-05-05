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


class TestAttention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.B = 3
        cls.L = 12
        cls.dm = 10
        cls.dk = 5
        cls.dv = 5

        key = jax.random.split(jax.random.PRNGKey(0), 5)
        cls.key = key[0]
        cls.Q = jax.random.normal(key[1], (cls.B, cls.L, cls.dm))
        cls.K = jax.random.normal(key[2], (cls.B, cls.L, cls.dm))
        cls.V = jax.random.normal(key[3], (cls.B, cls.L, cls.dm))

        cls.mask = jnp.asarray([[[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                                [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]],
                                [[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]]], dtype=int)
        assert cls.mask.shape == (cls.B, 1, cls.L), "BUG"

        cls.a = Attention(dk=cls.dk, dv=cls.dv)
        cls.params = cls.a.init(key[4], cls.Q, cls.K, cls.V, cls.mask)

    def test_without_mask_without_jit(self):
        A = self.a.apply(self.params, self.Q, self.K, self.V)
        self.assertEqual(A.shape, (self.B, self.L, self.dv))
        self.assertFalse(jnp.any(jnp.isnan(A)))


    def test_with_mask_without_jit(self):
        A = self.a.apply(self.params, self.Q, self.K, self.V, self.mask)
        self.assertEqual(A.shape, (self.B, self.L, self.dv))
        self.assertFalse(jnp.all(A == self.a.apply(self.params,
                                                   self.Q, self.K, self.V)))
        self.assertFalse(jnp.any(jnp.isnan(A)))

    def test_empty_mask_without_jit(self):
        """
        If ``mask`` is empty (aka. ``[0, 0, ..., 0]``),
        attention returns ``[nan, nan, ..., nan]`` because of softmax.

        In usual usecase, we must pass at lease one token like ``[BOS]``.
        """
        A = self.a.apply(self.params, self.Q, self.K, self.V,
                         jnp.zeros((self.B, 1, self.L), dtype=int))
        self.assertTrue(jnp.all(jnp.isnan(A)))

    def test_without_mask_with_jit(self):
        A = jax.jit(self.a.apply)(self.params, self.Q, self.K, self.V)
        self.assertEqual(A.shape, (self.B, self.L, self.dv))
        self.assertFalse(jnp.any(jnp.isnan(A)))

    def test_with_mask_with_jit(self):
        f = jax.jit(self.a.apply)
        A = f(self.params, self.Q, self.K, self.V, self.mask)
        self.assertEqual(A.shape, (self.B, self.L, self.dv))
        self.assertFalse(jnp.all(A == f(self.params, self.Q, self.K, self.V)))
        self.assertFalse(jnp.any(jnp.isnan(A)))

    def test_empty_mask_with_jit(self):
        A = jax.jit(self.a.apply)(self.params, self.Q, self.K, self.V,
                                  jnp.zeros((self.B, 1, self.L), dtype=int))
        self.assertTrue(jnp.all(jnp.isnan(A)))


class TestMultiHeadAttention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.B = 3
        cls.L = 12
        cls.dm = 10
        cls.nH = 5
        cls.Pdrop = 0.8

        key = jax.random.split(jax.random.PRNGKey(0), 5)
        cls.key = key[0]
        cls.Q = jax.random.normal(key[1], (cls.B, cls.L, cls.dm))
        cls.K = jax.random.normal(key[2], (cls.B, cls.L, cls.dm))
        cls.V = jax.random.normal(key[3], (cls.B, cls.L, cls.dm))

        cls.mask = jnp.asarray([[[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                                [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]],
                                [[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]]], dtype=int)
        assert cls.mask.shape == (cls.B, 1, cls.L), "BUG"

        cls.a = MultiHeadAttention(nH=cls.nH, dm=cls.dm, Pdrop=cls.Pdrop)
        cls.params = cls.a.init(key[4], cls.Q, cls.K, cls.V, cls.mask)

    def test_mha(self):
        f = self.a.apply
        f_jit = jax.jit(f)

        A = f(self.params, self.Q, self.K, self.V)
        self.assertEqual(A.shape, (self.B, self.L, self.dm))
        self.assertFalse(jnp.any(jnp.isnan(A)))

        A_jit = f_jit(self.params, self.Q, self.K, self.V)
        self.assertEqual(A_jit.shape, (self.B, self.L, self.dm))
        self.assertFalse(jnp.any(jnp.isnan(A_jit)))

        self.assertTrue(jnp.allclose(A, A_jit, atol=1e-6))

        A_mask = f(self.params, self.Q, self.K, self.V, self.mask)
        self.assertEqual(A_mask.shape, (self.B, self.L, self.dm))
        self.assertFalse(jnp.any(jnp.isnan(A_mask)))

        self.assertFalse(jnp.allclose(A, A_mask, atol=1e-5))

        A_mask_jit = f_jit(self.params, self.Q, self.K, self.V, self.mask)
        self.assertEqual(A_mask_jit.shape, (self.B, self.L, self.dm))
        self.assertFalse(jnp.any(jnp.isnan(A_mask_jit)))

        self.assertTrue(jnp.allclose(A_mask, A_mask_jit, atol=1e-6))
        self.assertFalse(jnp.allclose(A_jit, A_mask_jit, atol=1e-5))

if __name__ == "__main__":
    unittest.main()
