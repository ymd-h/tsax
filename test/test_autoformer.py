import unittest

import jax
import jax.numpy as jnp

from tsax.testing import TestCase
from tsax.model.autoformer import (
    Autoformer,
    EncoderStack,
    DecoderStack,
    EncoderLayer,
    DecoderLayer,
    MultiHeadAttention,
    AutoCorrelationAttention,
    SeasonalLayerNorm,
    SeriesDecomp,
)


class TestSeriesDecomp(TestCase):
    def test_sin(self):
        x = jnp.reshape(jnp.sin(0.5 * jnp.arange(12) * jnp.pi), (1, -1, 1))
        (s, t), _ = SeriesDecomp(kMA=4).init_with_output(jax.random.PRNGKey(0), x)

        self.assertEqual(x.shape, s.shape)
        self.assertEqual(x.shape, t.shape)
        self.assertAllclose(t, 0, atol=1e-6, rtol=1e-6)
        self.assertAllclose(s, x, atol=1e-6, rtol=1e-6)



if __name__ == "__main__":
    unittest.main()
