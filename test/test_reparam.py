import unittest

import jax
import jax.numpy as jnp

from tsax.testing import TestCase
from tsax.core.reparam import SigmaReparamDense


class TestSigmaReparamDense(TestCase):
    def test_reparam(self):
        features = 3
        B, L, d = 2, 7, 5

        key = jax.random.PRNGKey(42)

        key, key_use = jax.random.split(key)
        x = jax.random.normal(key_use, (B, L, d))

        key, key_use = jax.random.split(key)
        D = SigmaReparamDense(features)

        y, p = D.init_with_output(key_use, x)
        self.assertEqual(y.shape, (B, L, features))
        self.assertIn("sigma_reparam", p)

        y_jit, p_jit = jax.jit(D.init_with_output)(key_use, x)
        self.assertEqual(y_jit.shape, (B, L, features))
        self.assertIn("sigma_reparam", p_jit)

        self.assertAllclose(y, y_jit)


        y_train, p_train = D.init_with_output(key_use, x, train=True)
        self.assertEqual(y_train.shape, (B, L, features))
        self.assertIn("sigma_reparam", p_train)
        self.assertNotAllclose(y_train, y)

        y_train_jit, p_train_jit = jax.jit(D.init_with_output,
                                           static_argnames=["train"])(
                                               key_use, x, train=True
                                           )
        self.assertEqual(y_train_jit.shape, (B, L, features))
        self.assertIn("sigma_reparam", p_train_jit)
        self.assertNotAllclose(y_train_jit, y_jit)

        self.assertAllclose(y_train, y_train_jit)


if __name__ == "__main__":
    unittest.main()
