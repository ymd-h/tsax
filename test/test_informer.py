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
    FeedForward,
)
from tsax.testing import TestCase


class TestFeedForward(TestCase):
    def test_ff(self):
        B, L, dm = 2, 12, 5
        dff = 32
        Pdrop = 0.8

        key = jax.random.PRNGKey(0)

        key, key_use = jax.random.split(key, 2)
        x = jax.random.normal(key_use, (B, L, dm))

        FF = FeedForward(dff=dff, Pdrop=Pdrop)

        key_p, key_d = jax.random.split(key, 2)

        ff, _ = FF.init_with_output(key_p, x)
        self.assertEqual(ff.shape, x.shape)

        ff_jit, _ = jax.jit(FF.init_with_output)(key_p, x)
        self.assertEqual(ff_jit.shape, x.shape)

        self.assertAllclose(ff, ff_jit)

        ff_drop, _ = FF.init_with_output({"params": key_p, "dropout": key_d},
                                         x, with_dropout=True)
        self.assertEqual(ff_drop.shape, x.shape)
        self.assertNotAllclose(ff, ff_drop)

        ff_drop_jit, _ = jax.jit(
            FF.init_with_output,
            static_argnames=["with_dropout"],
        )({"params": key_p, "dropout": key_d}, x, with_dropout=True)
        self.assertEqual(ff_drop_jit.shape, x.shape)
        self.assertNotAllclose(ff_jit, ff_drop_jit)

        self.assertAllclose(ff_drop, ff_drop_jit)

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

        A = ProbSparseAttention(c=1, dk=d, dv=d, mask=3)

        key, key_use = jax.random.split(key, 2)
        a, _ = A.init_with_output({"params": key_use, "attention": key},
                                  Q, K, V)
        self.assertEqual(a.shape, Q.shape)
        self.assertNone(a, jnp.isnan)

        a_jit, _ = jax.jit(A.init_with_output)({"params": key_use,
                                                "attention": key},
                                               Q, K, V)
        self.assertEqual(a_jit.shape, Q.shape)
        self.assertNone(a_jit, jnp.isnan)

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
                                 mask=0)

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
                                 mask=6)

        key, key_use = jax.random.split(key, 2)
        mha, _ = MHA.init_with_output({"params": key_use, "attention": key},
                                      Q, K, V)
        self.assertEqual(mha.shape, Q.shape)

        mha_jit, _ = jax.jit(MHA.init_with_output)({"params": key_use,
                                                    "attention": key},
                                                   Q, K, V)
        self.assertEqual(mha_jit.shape, Q.shape)

        self.assertAllclose(mha, mha_jit)


class TestEncoderLayer(TestCase):
    def test_encoder(self):
        B, L, dm = 2, 12, 4
        c = 2
        nH = 2
        dff = 32
        eps = 0.1
        Pdrop = 0.8

        key = jax.random.PRNGKey(0)

        key, key_use = jax.random.split(key, 2)
        inputs = jax.random.normal(key_use, (B, L, dm))

        E = EncoderLayer(c=c, nH=nH, dm=dm, dff=dff, eps=eps, Pdrop=Pdrop)

        key_p, key_a, key_d = jax.random.split(key, 3)
        e, _ = E.init_with_output({"params": key_p,
                                   "attention": key_a,
                                   "dropout": key_d},
                                  inputs)
        self.assertEqual(e.shape, inputs.shape)

        e_jit, _ = jax.jit(E.init_with_output)({"params": key_p,
                                                "attention": key_a,
                                                "dropout": key_d},
                                               inputs)
        self.assertEqual(e_jit.shape, inputs.shape)

        self.assertAllclose(e, e_jit)

        e_drop, _ = E.init_with_output({"params": key_p,
                                        "attention": key_a,
                                        "dropout": key_d},
                                       inputs, with_dropout=True)
        self.assertEqual(e_drop.shape, inputs.shape)
        self.assertNotAllclose(e, e_drop)

        e_drop_jit, _ = jax.jit(
            E.init_with_output,
            static_argnames=["with_dropout"],
        )({"params": key_p, "attention": key_a, "dropout": key_d},
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
        eps = 0.1
        Pdrop = 0.8

        key = jax.random.PRNGKey(0)

        key, key_use = jax.random.split(key, 2)
        inputs = jax.random.normal(key_use, (B, L, dm))

        key, key_use = jax.random.split(key, 2)
        outputs = jax.random.normal(key_use, (B, L, dm))

        D = DecoderLayer(c=c, nH=nH, dm=dm, dff=dff, eps=eps, Pdrop=Pdrop, Ltoken=6)

        key_p, key_a, key_d = jax.random.split(key, 3)
        d, _ = D.init_with_output({"params": key_p,
                                   "attention": key_a,
                                   "dropout": key_d},
                                  inputs, outputs)
        self.assertEqual(d.shape, outputs.shape)

        d_jit, _ = jax.jit(D.init_with_output)({"params": key_p,
                                                "attention": key_a,
                                                "dropout": key_d},
                                               inputs, outputs)
        self.assertEqual(d_jit.shape, outputs.shape)

        self.assertAllclose(d, d_jit, atol=1e-6)

        d_drop, _ = D.init_with_output({"params": key_p,
                                        "attention": key_a,
                                        "dropout": key_d},
                                       inputs, outputs, with_dropout=True)
        self.assertEqual(d_drop.shape, outputs.shape)
        self.assertNotAllclose(d, d_drop)

        d_drop_jit, _ = jax.jit(
            D.init_with_output,
            static_argnames=["with_dropout"],
        )({"params": key_p, "attention": key_a, "dropout": key_d},
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
        kernel = 3
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
                         kernel=kernel,
                         eps=eps,
                         Pdrop=Pdrop)

        key_p, key_a, key_d = jax.random.split(key, 3)
        rngs = {
            "params": key_p,
            "attention": key_a,
            "dropout": key_d,
        }

        e, _ = E.init_with_output(rngs, inputs)
        self.assertEqual(e.shape, (B, ((L+1)//2 + 1)//2, dm))

        e_jit, _ = jax.jit(E.init_with_output)(rngs, inputs)
        self.assertEqual(e_jit.shape, (B, ((L+1)//2 + 1)//2, dm))

        self.assertAllclose(e, e_jit)

        e_drop, _ = E.init_with_output(rngs, inputs, with_dropout=True)
        self.assertEqual(e_drop.shape, (B, ((L+1)//2 + 1)//2, dm))
        self.assertNotAllclose(e, e_drop)

        e_drop_jit, _ = jax.jit(
            E.init_with_output,
            static_argnames=["with_dropout"],
        )(rngs, inputs, with_dropout=True)
        self.assertEqual(e_drop_jit.shape, (B, ((L+1)//2 + 1)//2, dm))
        self.assertNotAllclose(e_jit, e_drop_jit)

        self.assertAllclose(e_drop, e_drop_jit)

class TestDecoderStack(TestCase):
    def test_decoder(self):
        B, Lenc, Ldec, dm = 2, 4, 12, 6
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

        key_p, key_a, key_d = jax.random.split(key, 3)
        rngs = {
            "params": key_p,
            "attention": key_a,
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


class TestInformer(TestCase):
    def test_without_categorical(self):
        B, I, d, dm = 1, 5, 2, 8
        O = 5
        L = 2
        c = 2
        nE, nD, nH = 2, 3, 4
        dff = 16
        kernel = 3
        eps = 1e-8
        Pdrop = 0.8

        key = jax.random.PRNGKey(0)

        key, key_use = jax.random.split(key, 2)
        seq = jax.random.normal(key_use, (B, I, d))

        info = Informer(d=d,
                        I=I,
                        O=O,
                        Ltoken=L,
                        dm=dm,
                        c=c,
                        nE=nE,
                        nD=nD,
                        nH=nH,
                        dff=dff,
                        kernel=kernel,
                        eps=eps,
                        Pdrop=Pdrop)

        key_p, key_a, key_d = jax.random.split(key, 3)
        rngs = {
            "params": key_p,
            "attention": key_a,
            "dropout": key_d,
        }

        enc, _ = info.init_with_output(rngs, seq, method="encode")
        enc_jit, _ = jax.jit(
            info.init_with_output,
            static_argnames=["method"],
        )(rngs, seq, method="encode")
        self.assertAllclose(enc, enc_jit)

        pred, _ = info.init_with_output(rngs, seq)
        self.assertEqual(pred.shape, (B, O, d))

        pred_jit, _ = jax.jit(info.init_with_output)(rngs, seq)
        self.assertEqual(pred_jit.shape, (B, O, d))

        self.assertAllclose(pred, pred_jit, atol=1e-6)

        pred_drop, _ = info.init_with_output(rngs, seq, train=True)
        self.assertEqual(pred_drop.shape, (B, O, d))
        self.assertNotAllclose(pred, pred_drop)

        pred_drop_jit, _ = jax.jit(
            info.init_with_output,
            static_argnames=["train"],
        )(rngs, seq, train=True)
        self.assertEqual(pred_drop_jit.shape, (B, O, d))
        self.assertNotAllclose(pred_jit, pred_drop_jit)

        self.assertAllclose(pred_drop, pred_drop_jit, atol=1e-6)

    def test_with_categorical(self):
        B, I, d, Vs, dm = 1, 5, 2, (7, 12), 8
        O = 5
        L = 2
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

        info = Informer(d=d,
                        I=I,
                        O=O,
                        Ltoken=L,
                        dm=dm,
                        Vs=Vs,
                        c=c,
                        nE=nE,
                        nD=nD,
                        nH=nH,
                        dff=dff,
                        kernel=kernel,
                        eps=eps,
                        Pdrop=Pdrop)

        key_p, key_a, key_d = jax.random.split(key, 3)
        rngs = {
            "params": key_p,
            "attention": key_a,
            "dropout": key_d,
        }

        enc, _ = info.init_with_output(rngs, seq, cat, method="encode")
        enc_jit, _ = jax.jit(
            info.init_with_output,
            static_argnames=["method"],
        )(rngs, seq, cat, method="encode")
        self.assertAllclose(enc, enc_jit)

        pred, _ = info.init_with_output(rngs, seq, cat)
        self.assertEqual(pred.shape, (B, O, d))

        pred_jit, _ = jax.jit(info.init_with_output)(rngs, seq, cat)
        self.assertEqual(pred_jit.shape, (B, O, d))

        self.assertAllclose(pred, pred_jit, atol=5e-4, rtol=5e-4)

        pred_drop, _ = info.init_with_output(rngs, seq, cat, train=True)
        self.assertEqual(pred_drop.shape, (B, O, d))
        self.assertNotAllclose(pred, pred_drop)

        pred_drop_jit, _ = jax.jit(
            info.init_with_output,
            static_argnames=["train"],
        )(rngs, seq, cat, train=True)
        self.assertEqual(pred_drop_jit.shape, (B, O, d))
        self.assertNotAllclose(pred_jit, pred_drop_jit)

        self.assertAllclose(pred_drop, pred_drop_jit, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
