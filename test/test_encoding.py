import unittest

import jax
import jax.numpy as jnp

from tsax.core import PositionalEncoding
from tsax.testing import TestCase


class TestPositionalEncoding(TestCase):
    def test_even(self):
        L, dm, Lfreq = 2, 2, 2

        PE = PositionalEncoding(dm=dm, L=L, Lfreq=Lfreq)
        x = jnp.zeros((1, L, dm))

        pe = PE(x)

        self.assertAllclose(
            pe,
            jnp.asarray([
                [0, 1],
                [jnp.sin(1), jnp.cos(1)],
            ])
        )

    def test_odd(self):
        L, dm, Lfreq = 2, 1, 2

        PE = PositionalEncoding(dm=dm, L=L, Lfreq=Lfreq)
        x = jnp.zeros((1, L, dm))

        pe = PE(x)

        self.assertAllclose(
            pe,
            jnp.asarray([
                [0],
                [jnp.sin(1)],
            ])
        )

    def test_odd_larger(self):
        L, dm, Lfreq = 4, 3, 3

        PE = PositionalEncoding(dm=dm, L=L, Lfreq=Lfreq)
        x = jnp.zeros((4, L, dm))

        pe = PE(x)

        self.assertAllclose(
            pe,
            jnp.asarray([
                [0, 1, 0],
                [jnp.sin(1), jnp.cos(1), jnp.sin(1/(Lfreq ** (2 / dm)))],
                [jnp.sin(2), jnp.cos(2), jnp.sin(2/(Lfreq ** (2 / dm)))],
                [jnp.sin(3), jnp.cos(3), jnp.sin(3/(Lfreq ** (2 / dm)))],
            ])
        )

    def test_even_larger(self):
        L, dm, Lfreq = 4, 4, 3

        PE = PositionalEncoding(dm=dm, L=L, Lfreq=Lfreq)
        x = jnp.zeros((4, L, dm))

        pe = PE(x)

        self.assertAllclose(
            pe,
            jnp.asarray([
                [0, 1, 0, 1],
                [jnp.sin(1), jnp.cos(1),
                 jnp.sin(1/(Lfreq**(2/dm))), jnp.cos(1/(Lfreq**(2/dm)))],
                [jnp.sin(2), jnp.cos(2),
                 jnp.sin(2/(Lfreq**(2/dm))), jnp.cos(2/(Lfreq**(2/dm)))],
                [jnp.sin(3), jnp.cos(3),
                 jnp.sin(3/(Lfreq**(2/dm))), jnp.cos(3/(Lfreq**(2/dm)))],
            ])
        )

    def test_even_lazy(self):
        L, dm, Lfreq = 2, 2, 2

        PE = PositionalEncoding(dm=dm, L=L, Lfreq=Lfreq, lazy=True)
        x = jnp.zeros((1, L, dm))

        pe = PE(x)

        self.assertAllclose(
            pe,
            jnp.asarray([
                [0, 1],
                [jnp.sin(1), jnp.cos(1)],
            ])
        )

    def test_odd_lazy(self):
        L, dm, Lfreq = 2, 1, 2

        PE = PositionalEncoding(dm=dm, L=L, Lfreq=Lfreq, lazy=True)
        x = jnp.zeros((1, L, dm))

        pe = PE(x)

        self.assertAllclose(
            pe,
            jnp.asarray([
                [0],
                [jnp.sin(1)],
            ])
        )

if __name__ == "__main__":
    unittest.main()
