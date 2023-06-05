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

        ac, _ = AutoCorrelationAttention(dk=d, dv=d, c=c).init_with_output(
            jax.random.PRNGKey(42), Q, K, V
        )
        self.assertEqual(ac.shape, (*Q.shape[:2], d))

        ac_jit, _ = jax.jit(AutoCorrelationAttention(d=d, c=c).init_with_output)(
            jax.random.PRNGKey(42), Q, K, V
        )
        self.assertEqual(ac_jit.shape, (*Q.shape[:2], d))

        self.assertAllclose(ac, ac_jit)


class TestEncoderLayer(TestCase):
    def test_encoder(self):
        B, L, dm = 2, 12, 4
        c = 2
        nH = 2
        dff = 32
        Pdrop = 0.8

        key = jax.random.PRNGKey(0)

        key, key_use = jax.random.split(key, 2)
        inputs = jax.random.normal(key_use, (B, L, dm))

        E = EncoderLayer(c=c, nH=nH, dm=dm, dff=dff, Pdrop=Pdrop)

        key_p, key_d = jax.random.split(key, 2)
        e, _ = E.init_with_output({"params": key_p,
                                   "dropout": key_d},
                                  inputs)
        self.assertEqual(e.shape, inputs.shape)

        e_jit, _ = jax.jit(E.init_with_output)({"params": key_p,
                                                "dropout": key_d},
                                               inputs)
        self.assertEqual(e_jit.shape, inputs.shape)

        self.assertAllclose(e, e_jit)

        e_drop, _ = E.init_with_output({"params": key_p,
                                        "dropout": key_d},
                                       inputs, with_dropout=True)
        self.assertEqual(e_drop.shape, inputs.shape)
        self.assertNotAllclose(e, e_drop)

        e_drop_jit, _ = jax.jit(
            E.init_with_output,
            static_argnames=["with_dropout"],
        )({"params": key_p, "dropout": key_d},
          inputs, with_dropout=True)
        self.assertEqual(e_drop_jit.shape, inputs.shape)
        self.assertNotAllclose(e_jit, e_drop_jit)

        self.assertAllclose(e_drop, e_drop_jit)


class TestDecoderLayer(TestCase):
    def test_decoder(self):
        B, L, dm = 2, 12, 4
        c = 2
        nH = 2
        dff = 32
        Pdrop = 0.8

        key = jax.random.PRNGKey(0)

        key, key_use = jax.random.split(key, 2)
        inputs = jax.random.normal(key_use, (B, L, dm))

        key, key_use = jax.random.split(key, 2)
        outputs = jax.random.normal(key_use, (B, L, dm))

        D = DecoderLayer(c=c, nH=nH, dm=dm, dff=dff, eps=eps, Pdrop=Pdrop, Ltoken=6)

        key_p, key_d = jax.random.split(key, 2)
        d, _ = D.init_with_output({"params": key_p,
                                   "dropout": key_d},
                                  inputs, outputs)
        self.assertEqual(d.shape, outputs.shape)

        d_jit, _ = jax.jit(D.init_with_output)({"params": key_p,
                                                "dropout": key_d},
                                               inputs, outputs)
        self.assertEqual(d_jit.shape, outputs.shape)

        self.assertAllclose(d, d_jit, atol=1e-6)

        d_drop, _ = D.init_with_output({"params": key_p,
                                        "dropout": key_d},
                                       inputs, outputs, with_dropout=True)
        self.assertEqual(d_drop.shape, outputs.shape)
        self.assertNotAllclose(d, d_drop)

        d_drop_jit, _ = jax.jit(
            D.init_with_output,
            static_argnames=["with_dropout"],
        )({"params": key_p, "dropout": key_d},
          inputs, outputs, with_dropout=True)
        self.assertEqual(d_drop_jit.shape, outputs.shape)
        self.assertNotAllclose(d_jit, d_drop_jit)

        self.assertAllclose(d_drop, d_drop_jit, atol=1e-6)


class TestEncoderStack(TestCase):
    def test_encoder(self):
        B, L, dm = 2, 12, 6
        c = 3
        nE = 2
        nH = 3
        dff = 12
        eps = 1e-8
        Pdrop = 0.8

        key = jax.random.PRNGKey(0)

        key, key_use = jax.random.split(key, 2)
        inputs = jax.random.normal(key_use, (B, L, dm))

        E = EncoderStack(c=c,
                         dm=dm,
                         nE=nE,
                         nH=nH,
                         dff=dff,
                         eps=eps,
                         Pdrop=Pdrop)

        key_p, key_d = jax.random.split(key, 2)
        rngs = {
            "params": key_p,
            "dropout": key_d,
        }

        e, _ = E.init_with_output(rngs, inputs)
        self.assertEqual(e.shape, (B, (L+1)//2, dm))

        e_jit, _ = jax.jit(E.init_with_output)(rngs, inputs)
        self.assertEqual(e_jit.shape, (B, (L+1)//2, dm))

        self.assertAllclose(e, e_jit)

        e_drop, _ = E.init_with_output(rngs, inputs, with_dropout=True)
        self.assertEqual(e_drop.shape, (B, (L+1)//2, dm))
        self.assertNotAllclose(e, e_drop)

        e_drop_jit, _ = jax.jit(
            E.init_with_output,
            static_argnames=["with_dropout"],
        )(rngs, inputs, with_dropout=True)
        self.assertEqual(e_drop_jit.shape, (B, (L+1)//2, dm))
        self.assertNotAllclose(e_jit, e_drop_jit)

        self.assertAllclose(e_drop, e_drop_jit, atol=1e-6, rtol=1e-6)

class TestDecoderStack(TestCase):
    def test_decoder(self):
        B, Lenc, Ldec, dm = 2, 4, 4, 6
        c = 3
        nD = 2
        nH = 3
        dff = 12
        eps = 1e-8
        Pdrop = 0.8

        key = jax.random.PRNGKey(0)

        key, key_use = jax.random.split(key, 2)
        inputs = jax.random.normal(key_use, (B, Lenc, dm))

        key, key_use = jax.random.split(key, 2)
        outputs = jax.random.normal(key_use, (B, Ldec, dm))

        D = DecoderStack(c=c,
                         dm=dm,
                         nD=nD,
                         nH=nH,
                         dff=dff,
                         Ltoken=6,
                         eps=eps,
                         Pdrop=Pdrop)

        key_p, key_d = jax.random.split(key, 2)
        rngs = {
            "params": key_p,
            "dropout": key_d,
        }

        d, _ = D.init_with_output(rngs, inputs, outputs)
        self.assertEqual(d.shape, (B, Ldec, dm))

        d_jit, _ = jax.jit(D.init_with_output)(rngs, inputs, outputs)
        self.assertEqual(d_jit.shape, (B, Ldec, dm))

        self.assertAllclose(d, d_jit, atol=1e-6, rtol=1e-6)

        d_drop, _ = D.init_with_output(rngs, inputs, outputs, with_dropout=True)
        self.assertEqual(d_drop.shape, (B, Ldec, dm))
        self.assertNotAllclose(d, d_drop, atol=1e-6, rtol=1e-6)

        d_drop_jit, _ = jax.jit(
            D.init_with_output,
            static_argnames=["with_dropout"],
        )(rngs, inputs, outputs, with_dropout=True)
        self.assertEqual(d_drop_jit.shape, (B, Ldec, dm))
        self.assertNotAllclose(d_jit, d_drop_jit, atol=1e-6, rtol=1e-6)

        self.assertAllclose(d_drop, d_drop_jit, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
