from __future__ import annotations
from typing import Callable, Optional
import unittest

import jax.numpy as jnp

from tsax.typing import Array, ArrayLike

class TestCase(unittest.TestCase):
    def assertAllclose(self, x: ArrayLike, y: ArrayLike, **kwargs):
        self.assertTrue(jnp.allclose(x, y, **kwargs), msg=f"\nx: {x}\ny: {y}")

    def assertNotAllclose(self, x: ArrayLike, y: ArrayLike, **kwargs):
        self.assertFalse(jnp.allclose(x, y, **kwargs), msg=f"\nx: {x}\ny: {y}")

    def assertAll(self,
                  x: ArrayLike,
                  fn: Optional[Callable[[ArrayLike], Array]] = None):
        if fn is not None:
            _x = fn(x)
        else:
            _x = x

        self.assertTrue(jnp.all(_x), msg=f"\nx: {x}")

    def assertAny(self,
                  x: ArrayLike,
                  fn: Optional[Callable[[ArrayLike], Array]] = None):
        if fn is not None:
            _x = fn(x)
        else:
            _x = x

        self.assertTrue(jnp.any(_x), msg=f"\nx: {x}")

    def assertNone(self,
                   x: ArrayLike,
                   fn: Optional[Callable[[ArrayLike], Array]] = None):
        if fn is not None:
            _x = fn(x)
        else:
            _x = x

        self.assertFalse(jnp.any(_x), msg=f"\nx: {x}")
