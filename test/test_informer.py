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


if __name__ == "__main__":
    unittest.main()
