import unittest

import jax
import jax.numpy as jnp

from tsax.core import SubsequentMask
from tsax.testing import TestCase


class TestSubsequentMask(TestCase):
    def test_mask(self):
        self.assertAllclose(SubsequentMask(1),
                            jnp.asarray([[1]], dtype=int))


        self.assertAllclose(SubsequentMask(2),
                            jnp.asarray([[1, 0],
                                         [1, 1]], dtype=int))

        self.assertAllclose(SubsequentMask(3),
                            jnp.asarray([[1, 0, 0],
                                         [1, 1, 0],
                                         [1, 1, 1]], dtype=int))

if __name__ == "__main__":
    unittest.main()
