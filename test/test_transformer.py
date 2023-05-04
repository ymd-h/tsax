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
    def test_no_error(self):
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

    def test_no_error_jit(self):
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


if __name__ == "__main__":
    unittest.main()
