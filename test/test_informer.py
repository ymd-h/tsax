import unittest

import jax
import jax.numpy as jnp
from flax import linen as nn

from tsax.informer import (
    Informer,
    EncoderStack,
    DecoderStack,
    EncoderLayer,
    DecoderLayer,
    Distilling,
    MultiHeadAttention,
    ProbSparseAttention,
    FeedForward,
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



if __name__ == "__main__":
    unittest.main()
