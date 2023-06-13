from __future__ import annotations
from typing import Callable, Optional
import unittest

import jax.numpy as jnp

from tsax.typing import Array, ArrayLike

class TestCase(unittest.TestCase):
    def assertAllclose(self, x: ArrayLike, y: ArrayLike, **kwargs):
        self.longMessage = False
        self.assertTrue(jnp.allclose(x, y, **kwargs),
                        msg=f"Arrays are not all close:\n" +
                        "x: {x}\ny: {y}\nmax|x-y|: {jnp.max(jnp.abs(x-y))}")
        self.longMessage = True

    def assertNotAllclose(self, x: ArrayLike, y: ArrayLike, **kwargs):
        self.longMessage = False
        self.assertEqual(x.shape, y.shape,
                         msg=f"Array Shape mismatch:\n{x.shape} != {y.shape}")
        self.assertFalse(jnp.allclose(x, y, **kwargs),
                         msg=f"Arrays are all close:\nx: {x}\ny: {y}")
        self.longMessage = True

    def assertAll(self,
                  x: ArrayLike,
                  fn: Optional[Callable[[ArrayLike], Array]] = None):
        _x: ArrayLike
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
