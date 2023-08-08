from __future__ import annotations
from typing import Callable, Optional
import unittest

import jax.numpy as jnp

from tsax.typing import Array, ArrayLike

__all__ = [
    "TestCase",
]

class TestCase(unittest.TestCase):
    def assertAllclose(self, x: ArrayLike, y: ArrayLike, **kwargs):
        x = jnp.asarray(x)
        y = jnp.asarray(y)
        self.longMessage = False
        self.assertTrue(jnp.allclose(x, y, **kwargs),
                        msg="Arrays are not all close:\n" +
                        f"\n{x} !=\n{y}\nmax|x-y|: {jnp.max(jnp.abs(x-y))}")
        self.longMessage = True

    def assertNotAllclose(self, x: ArrayLike, y: ArrayLike, **kwargs):
        x = jnp.asarray(x)
        y = jnp.asarray(y)
        self.longMessage = False
        self.assertEqual(x.shape, y.shape,
                         msg=f"Array Shape mismatch:\n{x.shape} != {y.shape}")
        self.assertFalse(jnp.allclose(x, y, **kwargs),
                         msg=f"Arrays are all close:\n{x} ==\n{y}")
        self.longMessage = True

    def assertAll(self,
                  x: ArrayLike,
                  fn: Optional[Callable[[ArrayLike], Array]] = None):
        _x: ArrayLike
        if fn is not None:
            _x = fn(x)
        else:
            _x = x

        self.assertTrue(jnp.all(_x), msg=f"\n{x}")

    def assertAny(self,
                  x: ArrayLike,
                  fn: Optional[Callable[[ArrayLike], Array]] = None):
        _x: ArrayLike
        if fn is not None:
            _x = fn(x)
        else:
            _x = x

        self.assertTrue(jnp.any(_x), msg=f"\n{x}")

    def assertNone(self,
                   x: ArrayLike,
                   fn: Optional[Callable[[ArrayLike], Array]] = None):
        _x: ArrayLike
        if fn is not None:
            _x = fn(x)
        else:
            _x = x

        self.assertFalse(jnp.any(_x), msg=f"\n{x}")
