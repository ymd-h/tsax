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
        self.assertAllclose(t.at[:,1:-2,:].get(), 0, atol=1e-6, rtol=1e-6)
        self.assertEqual(t.at[:,0,:].get(),
                         jnp.mean(x.at[:,(0,0,1,2),:].get(), axis=1))
        self.assertAllclose(s.at[:,1:-2,:].get(),
                            x.at[:,1:-2,:].get(), atol=1e-6, rtol=1e-6)


class TestSeasonalLayerNorm(TestCase):
    def test_seasonal(self):
        x = jnp.reshape(jnp.sin(0.5 * jnp.arange(12) * jnp.pi), (1, -1, 1))
        y, _ = SeasonalLayerNorm(eps=1e-8).init_with_output(jax.random.PRNGKey(0), x)

        self.assertEqual(x.shape, y.shape)
        self.assertAllclose(jnp.mean(y, axis=1), 0)


class TestAutoCrrelationAttention(TestCase):
    def test_ac(self):
        Q = jax.random.normal(jax.random.PRNGKey(0), (2, 10, 4))
        K = Q
        V = Q

        c = 2
        d = 2

        ac, _ = AutoCorrelationAttention(d=d, c=c).init_with_output(
            jax.random.PRNGKey(42), Q, K, V
        )
        self.assertEqual(ac.shape, (*Q.shape[:2], d))

        ac_jit, _ = jax.jit(AutoCorrelationAttention(d=d, c=c).init_with_output)(
            jax.random.PRNGKey(42), Q, K, V
        )
        self.assertEqual(ac_jit.shape, (*Q.shape[:2], d))

        self.assertAllclose(ac, ac_jit)

if __name__ == "__main__":
    unittest.main()
