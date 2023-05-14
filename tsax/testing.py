import unittest

import jax.numpy as jnp
from jax.typing import ArrayLike

class TestCase(unittest.TestCase):
    def assertAllclose(self, x: ArrayLike, y: ArrayLike, **kwargs):
        self.assertTrue(jnp.allclose(x, y, **kwargs), msg=f"\nx: {x}\ny: {y}")

    def assertNotAllclose(self, x: ArrayLike, y: ArrayLike, **kwargs):
        self.assertFalse(jnp.allclose(x, y, **kwargs), msg=f"\nx: {x}\ny: {y}")
