import unittest

import jax
import jax.numpy as jnp

from tsax.core import PositionalEncoding, CategoricalEncoding, Embedding
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

    def test_even_lazy_shift(self):
        L, dm, Lfreq = 2, 2, 2

        PE = PositionalEncoding(dm=dm, L=L, Lfreq=Lfreq, lazy=True)
        x = jnp.zeros((1, L, dm))

        pe = PE(x, shift = 2)

        self.assertAllclose(
            pe,
            jnp.asarray([
                [jnp.sin(2), jnp.cos(2)],
                [jnp.sin(3), jnp.cos(3)],
            ])
        )

    def test_odd_lazy_shift(self):
        L, dm, Lfreq = 2, 1, 2

        PE = PositionalEncoding(dm=dm, L=L, Lfreq=Lfreq, lazy=True)
        x = jnp.zeros((1, L, dm))

        pe = PE(x, shift = 2)

        self.assertAllclose(
            pe,
            jnp.asarray([
                [jnp.sin(2)],
                [jnp.sin(3)],
            ])
        )


class TestCategoricalEncoding(TestCase):
    def test_simple(self):
        V = 7
        dm = 3
        B = 1
        L = 2

        key = jax.random.PRNGKey(0)

        key, key_use = jax.random.split(key, 2)
        x = jax.random.randint(key_use, (B, L, 1), 0, V)

        e = CategoricalEncoding(Vs=(V,), dm=dm)

        key, key_use = jax.random.split(key, 2)
        embedded, _ = e.init_with_output(key_use, x)
        self.assertEqual(embedded.shape, (B, L, dm))

        embedded_jit, _ = jax.jit(e.init_with_output)(key_use, x)
        self.assertEqual(embedded_jit.shape, (B, L, dm))

        self.assertAllclose(embedded, embedded_jit)

    def test_multiple(self):
        Vs = (7, 31, 12)
        dm = 3
        B = 1
        L = 2

        key = jax.random.PRNGKey(0)

        key, key_use = jax.random.split(key, 2)
        x = jax.random.randint(key_use, (B, L, len(Vs)), 0, jnp.asarray(Vs))

        e = CategoricalEncoding(Vs=Vs, dm=dm)

        key, key_use = jax.random.split(key, 2)
        embedded, _ = e.init_with_output(key_use, x)
        self.assertEqual(embedded.shape, (B, L, dm))

        embedded_jit, _ = jax.jit(e.init_with_output)(key_use, x)
        self.assertEqual(embedded_jit.shape, (B, L, dm))

        self.assertAllclose(embedded, embedded_jit)


class TestEmbedding(TestCase):
    def test_without_categorical(self):
        B, L, d, dm = 1, 5, 2, 3

        key = jax.random.PRNGKey(0)

        key, key_use = jax.random.split(key, 2)
        x = jax.random.normal(key_use, (B, L, d))

        e = Embedding(dm=dm,
                      Vs=tuple(),
                      kernel=3,
                      alpha=1.0,
                      Pdrop=0.8,
                      with_positional=True)

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

        e = Embedding(dm=dm,
                      Vs=Vs,
                      kernel=3,
                      alpha=1.0,
                      Pdrop=0.8,
                      with_positional=True)

        key, key_use = jax.random.split(key, 2)
        embedded, _ = e.init_with_output(key_use, x, c)
        self.assertEqual(embedded.shape, (B, L, dm))

        embedded_jit, _ = jax.jit(e.init_with_output)(key_use, x, c)
        self.assertEqual(embedded_jit.shape, (B, L, dm))

        self.assertAllclose(embedded, embedded_jit)

        
if __name__ == "__main__":
    unittest.main()
