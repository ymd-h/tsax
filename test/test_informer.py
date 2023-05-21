import unittest

import jax
import jax.numpy as jnp
from flax import linen as nn

from tsax.model.informer import (
    Informer,
    EncoderStack,
    DecoderStack,
    EncoderLayer,
    DecoderLayer,
    Distilling,
    MultiHeadAttention,
    ProbSparseAttention,
    Embedding,
)
from tsax.testing import TestCase


class TestEmbedding(TestCase):
    def test_without_categorical(self):
        B, L, d, dm = 1, 5, 2, 3

        key = jax.random.PRNGKey(0)

        key, key_use = jax.random.split(key, 2)
        x = jax.random.normal(key_use, (B, L, d))

        e = Embedding(dm=dm)

        key, key_use = jax.random.split(key, 2)
        embedded, _ = e.init_with_output(key_use, x)
        self.assertEqual(embedded.shape, (B, L, dm))

        embedded_jit, _ = jax.jit(e.init_with_output)(key_use, x)
        self.assertEqual(embedded_jit.shape, (B, L, dm))

        self.assertAllclose(embedded, embedded_jit)

    def test_with_categorical(self):
        B, L, d, Vs, dm = 1, 5, 2, (7, 12), 3

        key = jax.random.PRNGKey(0)

        key, key_use = jax.random.split(key, 2)
        x = jax.random.normal(key_use, (B, L, d))

        key, key_use = jax.random.split(key, 2)
        c = jax.random.randint(key_use,
                               (B, L, len(Vs)),
                               0, jnp.asarray(Vs, dtype=int))

        e = Embedding(dm=dm, Vs=Vs)

        key, key_use = jax.random.split(key, 2)
        embedded, _ = e.init_with_output(key_use, x, c)
        self.assertEqual(embedded.shape, (B, L, dm))

        embedded_jit, _ = jax.jit(e.init_with_output)(key_use, x, c)
        self.assertEqual(embedded_jit.shape, (B, L, dm))

        self.assertAllclose(embedded, embedded_jit)


class TestDistilling(TestCase):
    def test_even(self):
        B, L, dm = 1, 4, 3

        key = jax.random.PRNGKey(0)

        key, key_use = jax.random.split(key, 2)
        x = jax.random.normal(key_use, (B, L, dm))

        d = Distilling(kernel=3)

        key, key_use = jax.random.split(key, 2)
        distilled, _ = d.init_with_output(key_use, x)
        self.assertEqual(distilled.shape, (B, (L+1)//2, dm))

        distilled_jit, _ = jax.jit(d.init_with_output)(key_use, x)
        self.assertEqual(distilled_jit.shape, (B, (L+1)//2, dm))

        self.assertAllclose(distilled, distilled_jit)

    def test_odd(self):
        B, L, dm = 1, 5, 3

        key = jax.random.PRNGKey(0)

        key, key_use = jax.random.split(key, 2)
        x = jax.random.normal(key_use, (B, L, dm))

        d = Distilling(kernel=3)

        key, key_use = jax.random.split(key, 2)
        distilled, _ = d.init_with_output(key_use, x)
        self.assertEqual(distilled.shape, (B, (L+1)//2, dm))

        distilled_jit, _ = jax.jit(d.init_with_output)(key_use, x)
        self.assertEqual(distilled_jit.shape, (B, (L+1)//2, dm))

        self.assertAllclose(distilled, distilled_jit)


class TestProbSparseAttention(TestCase):
    def test_attention(self):
        B, L, d = 1, 5, 3

        key = jax.random.PRNGKey(0)

        key, key_use = jax.random.split(key, 2)
        Q = jax.random.normal(key_use, (B, L, d))

        key, key_use = jax.random.split(key, 2)
        K = jax.random.normal(key_use, (B, L, d))

        key, key_use = jax.random.split(key, 2)
        V = jax.random.normal(key_use, (B, L, d))

        A = ProbSparseAttention(c=1, dk=d, dv=d)

        key, key_use = jax.random.split(key, 2)
        a, _ = A.init_with_output({"params": key_use, "attention": key},
                                  Q, K, V)
        self.assertEqual(a.shape, Q.shape)

        a_jit, _ = jax.jit(A.init_with_output)({"params": key_use,
                                                "attention": key}, Q, K, V)
        self.assertEqual(a_jit.shape, Q.shape)

        self.assertAllclose(a, a_jit)

    def test_attention_with_mask(self):
        B, L, d = 1, 5, 3

        key = jax.random.PRNGKey(0)

        key, key_use = jax.random.split(key, 2)
        Q = jax.random.normal(key_use, (B, L, d))

        key, key_use = jax.random.split(key, 2)
        K = jax.random.normal(key_use, (B, L, d))

        key, key_use = jax.random.split(key, 2)
        V = jax.random.normal(key_use, (B, L, d))

        A = ProbSparseAttention(c=1, dk=d, dv=d, mask=True)

        key, key_use = jax.random.split(key, 2)
        a, _ = A.init_with_output({"params": key_use, "attention": key},
                                  Q, K, V)
        self.assertEqual(a.shape, Q.shape)

        a_jit, _ = jax.jit(A.init_with_output)({"params": key_use,
                                                "attention": key},
                                               Q, K, V)
        self.assertEqual(a_jit.shape, Q.shape)

        self.assertAllclose(a, a_jit)


class TestMultiHeadAttention(TestCase):
    def test_mha(self):
        B, L, dm = 1, 12, 6
        c = 2
        nH = 2
        Pdrop = 0.8

        key = jax.random.PRNGKey(0)

        key, key_use = jax.random.split(key)
        Q = jax.random.normal(key_use, (B, L, dm))

        key, key_use = jax.random.split(key)
        K = jax.random.normal(key_use, (B, L, dm))

        key, key_use = jax.random.split(key)
        V = jax.random.normal(key_use, (B, L, dm))

        MHA = MultiHeadAttention(c=c,
                                 dm=dm,
                                 nH=nH,
                                 Pdrop=Pdrop,
                                 mask=False)

        key_p, key_a, key_d = jax.random.split(key, 3)
        mha, _ = MHA.init_with_output({"params": key_p, "attention": key_a},
                                      Q, K, V)
        self.assertEqual(mha.shape, Q.shape)

        mha_jit, _ = jax.jit(MHA.init_with_output)({"params": key_p,
                                                    "attention": key_a},
                                                   Q, K, V)
        self.assertEqual(mha_jit.shape, Q.shape)

        self.assertAllclose(mha, mha_jit)

        mha_drop, _ = MHA.init_with_output({"params": key_p,
                                            "attention": key_a,
                                            "dropout": key_d},
                                           Q, K, V, with_dropout=True)
        self.assertEqual(mha_drop.shape, Q.shape)
        self.assertNotAllclose(mha, mha_drop)

        mha_drop_jit, _ = jax.jit(
            MHA.init_with_output,
            static_argnames=["with_dropout"]
        )({"params": key_p, "attention": key_a, "dropout": key_d},
          Q, K, V, with_dropout=True)
        self.assertEqual(mha_drop_jit.shape, Q.shape)
        self.assertNotAllclose(mha_jit, mha_drop_jit)

        self.assertAllclose(mha_drop, mha_drop_jit)

    def test_mha_mask(self):
        B, L, dm = 1, 12, 6
        c = 2
        nH = 2
        Pdrop = 0.8

        key = jax.random.PRNGKey(0)

        key, key_use = jax.random.split(key)
        Q = jax.random.normal(key_use, (B, L, dm))

        key, key_use = jax.random.split(key)
        K = jax.random.normal(key_use, (B, L, dm))

        key, key_use = jax.random.split(key)
        V = jax.random.normal(key_use, (B, L, dm))

        MHA = MultiHeadAttention(c=c,
                                 dm=dm,
                                 nH=nH,
                                 Pdrop=Pdrop,
                                 mask=True)

        key, key_use = jax.random.split(key, 2)
        mha, _ = MHA.init_with_output({"params": key_use, "attention": key},
                                      Q, K, V)
        self.assertEqual(mha.shape, Q.shape)

        mha_jit, _ = jax.jit(MHA.init_with_output)({"params": key_use,
                                                    "attention": key},
                                                   Q, K, V)
        self.assertEqual(mha_jit.shape, Q.shape)

        self.assertAllclose(mha, mha_jit)

if __name__ == "__main__":
    unittest.main()
