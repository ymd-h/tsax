import unittest
import functools

import jax
import jax.numpy as jnp

from tsax.testing import TestCase
from tsax.core import MultiHeadAttention
from tsax.model.informer import (
    Attention,
    ProbSparseAttention,
)
from tsax.model.autoformer import (
    AutoCorrelationAttention,
)


class TestMultiHeadAttention(TestCase):
    def test_full(self):
        B, L, dm = 1, 12, 6
        nH = 2
        Pdrop = 0.8

        key = jax.random.PRNGKey(0)

        key, key_use = jax.random.split(key)
        Q = jax.random.normal(key_use, (B, L, dm))

        key, key_use = jax.random.split(key)
        K = jax.random.normal(key_use, (B, L, dm))

        key, key_use = jax.random.split(key)
        V = jax.random.normal(key_use, (B, L, dm))

        MHA = MultiHeadAttention(
            attention=Attention,
            dm=dm,
            nH=nH,
            Pdrop=Pdrop,
        )

        key_p, key_d = jax.random.split(key, 2)
        mha, _ = MHA.init_with_output({"params": key_p},
                                      Q, K, V)
        self.assertEqual(mha.shape, Q.shape)

        mha_jit, _ = jax.jit(MHA.init_with_output)({"params": key_p},
                                                   Q, K, V)
        self.assertEqual(mha_jit.shape, Q.shape)

        self.assertAllclose(mha, mha_jit, atol=1e-6, rtol=1e-6)

        mha_drop, _ = MHA.init_with_output({"params": key_p, "dropout": key_d},
                                           Q, K, V, with_dropout=True)
        self.assertEqual(mha_drop.shape, Q.shape)
        self.assertNotAllclose(mha, mha_drop)

        mha_drop_jit, _ = jax.jit(
            MHA.init_with_output,
            static_argnames=["with_dropout"]
        )({"params": key_p, "dropout": key_d},
          Q, K, V, with_dropout=True)
        self.assertEqual(mha_drop_jit.shape, Q.shape)
        self.assertNotAllclose(mha_jit, mha_drop_jit)

        self.assertAllclose(mha_drop, mha_drop_jit, atol=1e-6, rtol=1e-6)

    
    def test_probsparse(self):
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

        MHA = MultiHeadAttention(
            attention=functools.partial(ProbSparseAttention, c=c, mask=0),
            dm=dm,
            nH=nH,
            Pdrop=Pdrop,
        )

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

        MHA = MultiHeadAttention(
            attention=functools.partial(ProbSparseAttention, c=c, mask=6),
            dm=dm,
            nH=nH,
            Pdrop=Pdrop,
        )

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
