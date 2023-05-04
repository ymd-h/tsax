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
        key = jax.random.PRNGKey(0)
        text = jnp.zeros((4, 12), dtype=int)

        e = Embedding(V=5, L=12)
        params = e.init(key, text)

        ret = e.apply(params, text)

    def test_no_error_jit(self):
        key = jax.random.PRNGKey(0)
        text = jnp.zeros((4, 12), dtype=int)

        e = Embedding(V=5, L=12)
        params = e.init(key, text)

        ret = jax.jit(e.apply)(params, text)


if __name__ == "__main__":
    unittest.main()
