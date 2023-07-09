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
        d = 3
        c = 2

        Q = jax.random.normal(jax.random.PRNGKey(0), (2, 10, d))
        K = Q
        V = Q


        A = AutoCorrelationAttention(dk=d, dv=d, c=c)

        ac, _ = A.init_with_output(
            jax.random.PRNGKey(42), Q, K, V
        )
        self.assertEqual(ac.shape, (*Q.shape[:2], d))

        ac_jit, _ = jax.jit(A.init_with_output)(
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

        self.assertAllclose(e_drop, e_drop_jit, atol=1e-5, rtol=1e-5)


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
        s_outputs = jax.random.normal(key_use, (B, L, dm))

        key, key_use = jax.random.split(key, 2)
        t_outputs = jax.random.normal(key_use, (B, L, dm))

        D = DecoderLayer(c=c, nH=nH, dm=dm, dff=dff, Pdrop=Pdrop)

        key_p, key_d = jax.random.split(key, 2)
        (ds, dt), _ = D.init_with_output({"params": key_p,
                                          "dropout": key_d},
                                         inputs, s_outputs, t_outputs)
        self.assertEqual(ds.shape, s_outputs.shape)
        self.assertEqual(dt.shape, t_outputs.shape)

        (ds_jit, dt_jit), _ = jax.jit(D.init_with_output)({"params": key_p,
                                                           "dropout": key_d},
                                                          inputs,
                                                          s_outputs, t_outputs)
        self.assertEqual(ds_jit.shape, s_outputs.shape)
        self.assertEqual(dt_jit.shape, t_outputs.shape)

        self.assertAllclose(ds, ds_jit, atol=1e-6)
        self.assertAllclose(dt, dt_jit, atol=1e-6)

        (ds_drop, dt_drop), _ = D.init_with_output({"params": key_p,
                                                    "dropout": key_d},
                                                   inputs, s_outputs, t_outputs,
                                                   with_dropout=True)
        self.assertEqual(ds_drop.shape, s_outputs.shape)
        self.assertEqual(dt_drop.shape, t_outputs.shape)
        self.assertNotAllclose(ds, ds_drop)
        self.assertAllclose(dt, dt_drop)

        (ds_drop_jit, dt_drop_jit), _ = jax.jit(
            D.init_with_output,
            static_argnames=["with_dropout"],
        )({"params": key_p, "dropout": key_d},
          inputs, s_outputs, t_outputs, with_dropout=True)
        self.assertEqual(ds_drop_jit.shape, s_outputs.shape)
        self.assertEqual(dt_drop_jit.shape, t_outputs.shape)
        self.assertNotAllclose(ds_jit, ds_drop_jit)
        self.assertAllclose(dt_jit, dt_drop_jit)

        self.assertAllclose(ds_drop, ds_drop_jit, atol=1e-6)
        self.assertAllclose(dt_drop, dt_drop_jit, atol=1e-6)


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
        self.assertEqual(e.shape, (B, L, dm))

        e_jit, _ = jax.jit(E.init_with_output)(rngs, inputs)
        self.assertEqual(e_jit.shape, (B, L, dm))

        self.assertAllclose(e, e_jit, atol=1e-5, rtol=1e-5)

        e_drop, _ = E.init_with_output(rngs, inputs, with_dropout=True)
        self.assertEqual(e_drop.shape, (B, L, dm))
        self.assertNotAllclose(e, e_drop)

        e_drop_jit, _ = jax.jit(
            E.init_with_output,
            static_argnames=["with_dropout"],
        )(rngs, inputs, with_dropout=True)
        self.assertEqual(e_drop_jit.shape, (B, L, dm))
        self.assertNotAllclose(e_jit, e_drop_jit, atol=1e-6, rtol=1e-6)

        self.assertAllclose(e_drop, e_drop_jit, atol=1e-5, rtol=1e-5)

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
        s_outputs = jax.random.normal(key_use, (B, Ldec, dm))

        key, key_use = jax.random.split(key, 2)
        t_outputs = jax.random.normal(key_use, (B, Ldec, dm))

        D = DecoderStack(c=c,
                         dm=dm,
                         nD=nD,
                         nH=nH,
                         dff=dff,
                         eps=eps,
                         Pdrop=Pdrop)

        key_p, key_d = jax.random.split(key, 2)
        rngs = {
            "params": key_p,
            "dropout": key_d,
        }

        (ds, dt), _ = D.init_with_output(rngs, inputs, s_outputs, t_outputs)
        self.assertEqual(ds.shape, (B, Ldec, dm))
        self.assertEqual(dt.shape, (B, Ldec, dm))

        (ds_jit, dt_jit), _ = jax.jit(D.init_with_output)(rngs, inputs,
                                                          s_outputs, t_outputs)
        self.assertEqual(ds_jit.shape, (B, Ldec, dm))
        self.assertEqual(dt_jit.shape, (B, Ldec, dm))

        self.assertAllclose(ds, ds_jit, atol=1e-6, rtol=1e-6)
        self.assertAllclose(dt, dt_jit, atol=1e-6, rtol=1e-6)

        (ds_drop, dt_drop), _ = D.init_with_output(rngs, inputs, s_outputs, t_outputs,
                                                   with_dropout=True)
        self.assertEqual(ds_drop.shape, (B, Ldec, dm))
        self.assertEqual(dt_drop.shape, (B, Ldec, dm))
        self.assertNotAllclose(ds, ds_drop, atol=1e-6, rtol=1e-6)
        self.assertAllclose(dt, dt_drop, atol=1e-6, rtol=1e-6)

        (ds_drop_jit, dt_drop_jit), _ = jax.jit(
            D.init_with_output,
            static_argnames=["with_dropout"],
        )(rngs, inputs, s_outputs, t_outputs, with_dropout=True)
        self.assertEqual(ds_drop_jit.shape, (B, Ldec, dm))
        self.assertEqual(dt_drop_jit.shape, (B, Ldec, dm))
        self.assertNotAllclose(ds_jit, ds_drop_jit, atol=1e-6, rtol=1e-6)
        self.assertAllclose(dt_jit, dt_drop_jit, atol=1e-6, rtol=1e-6)

        self.assertAllclose(ds_drop, ds_drop_jit, atol=1e-5, rtol=1e-5)
        self.assertAllclose(dt_drop, dt_drop_jit, atol=1e-5, rtol=1e-5)

class TestAutoformer(TestCase):
    def test_without_categorical(self):
        B, I, d, dm = 1, 5, 2, 8
        O = 5
        c = 2
        nE, nD, nH = 2, 3, 4
        dff = 16
        kernel = 3
        eps = 1e-8
        Pdrop = 0.8

        key = jax.random.PRNGKey(0)

        key, key_use = jax.random.split(key, 2)
        seq = jax.random.normal(key_use, (B, I, d))

        auto = Autoformer(d=d,
                          I=I,
                          O=O,
                          c=c,
                          nE=nE,
                          nD=nD,
                          nH=nH,
                          dff=dff,
                          eps=eps,
                          Pdrop=Pdrop)

        key_p, key_a, key_d = jax.random.split(key, 3)
        rngs = {
            "params": key_p,
            "attention": key_a,
            "dropout": key_d,
        }

        enc, _ = auto.init_with_output(rngs, seq, method="encode")
        enc_jit, _ = jax.jit(
            auto.init_with_output,
            static_argnames=["method"],
        )(rngs, seq, method="encode")
        self.assertAllclose(enc, enc_jit, atol=1e-4, rtol=1e-4)

        pred, _ = auto.init_with_output(rngs, seq)
        self.assertEqual(pred.shape, (B, O, d))

        pred_jit, _ = jax.jit(auto.init_with_output)(rngs, seq)
        self.assertEqual(pred_jit.shape, (B, O, d))

        self.assertAllclose(pred, pred_jit, atol=1e-4, rtol=1e-4)

        pred_drop, _ = auto.init_with_output(rngs, seq, train=True)
        self.assertEqual(pred_drop.shape, (B, O, d))
        self.assertNotAllclose(pred, pred_drop)

        pred_drop_jit, _ = jax.jit(
            auto.init_with_output,
            static_argnames=["train"],
        )(rngs, seq, train=True)
        self.assertEqual(pred_drop_jit.shape, (B, O, d))
        self.assertNotAllclose(pred_jit, pred_drop_jit)

        self.assertAllclose(pred_drop, pred_drop_jit, atol=1e-6)

    def test_with_categorical(self):
        B, I, d, Vs, dm = 1, 5, 2, (7, 12), 8
        O = 5
        c = 2
        nE, nD, nH = 2, 3, 4
        dff = 16
        kernel = 3
        eps = 1e-8
        Pdrop = 0.8

        key = jax.random.PRNGKey(0)

        key, key_use = jax.random.split(key, 2)
        seq = jax.random.normal(key_use, (B, I, d))

        key, key_use = jax.random.split(key, 2)
        cat = jax.random.randint(key_use,
                                 (B, I, len(Vs)),
                                 0, jnp.asarray(Vs, dtype=int))

        auto = Autoformer(d=d,
                          I=I,
                          O=O,
                          Vs=Vs,
                          c=c,
                          nE=nE,
                          nD=nD,
                          nH=nH,
                          dff=dff,
                          eps=eps,
                          Pdrop=Pdrop)

        key_p, key_a, key_d = jax.random.split(key, 3)
        rngs = {
            "params": key_p,
            "attention": key_a,
            "dropout": key_d,
        }

        enc, _ = auto.init_with_output(rngs, seq, cat, method="encode")
        enc_jit, _ = jax.jit(
            auto.init_with_output,
            static_argnames=["method"],
        )(rngs, seq, cat, method="encode")
        self.assertAllclose(enc, enc_jit)

        pred, _ = auto.init_with_output(rngs, seq, cat)
        self.assertEqual(pred.shape, (B, O, d))

        pred_jit, _ = jax.jit(auto.init_with_output)(rngs, seq, cat)
        self.assertEqual(pred_jit.shape, (B, O, d))

        self.assertAllclose(pred, pred_jit, atol=5e-4, rtol=5e-4)

        pred_drop, _ = auto.init_with_output(rngs, seq, cat, train=True)
        self.assertEqual(pred_drop.shape, (B, O, d))
        self.assertNotAllclose(pred, pred_drop)

        pred_drop_jit, _ = jax.jit(
            auto.init_with_output,
            static_argnames=["train"],
        )(rngs, seq, cat, train=True)
        self.assertEqual(pred_drop_jit.shape, (B, O, d))
        self.assertNotAllclose(pred_jit, pred_drop_jit)

        self.assertAllclose(pred_drop, pred_drop_jit, atol=1e-4, rtol=1e-4)

if __name__ == "__main__":
    unittest.main()
