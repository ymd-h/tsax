import unittest

import jax
import jax.numpy as jnp

from tsax.testing import TestCase
from tsax.data import SeqData


class TestSeqDat(TestCase):
    def test_ibatch(self):
        d = jnp.reshape(jnp.arange(10, dtype=float), (-1, 1))

        seq = SeqData(d, xLen=1, yLen=1, batch_size=2)

        x, y = seq.ibatch(0)
        self.assertEqual(x.shape, (2, 1, 1))
        self.assertEqual(y.shape, (2, 1, 1))

        self.assertAllclose(x, jnp.asarray([[[0]], [[1]]]))
        self.assertAllclose(y, jnp.asarray([[[1]], [[2]]]))

    def test_ibatch_with_cat(self):
        d = jnp.reshape(jnp.arange(10, dtype=float), (-1, 1))
        c = jnp.reshape(jnp.arange(10), (-1, 1)) + 10

        seq = SeqData((d, c), xLen=1, yLen=2, batch_size=2)
        (xseq, xcat), (yseq, ycat) = seq.ibatch(0)

        self.assertEqual(xseq.shape, (2, 1, 1))
        self.assertEqual(xcat.shape, (2, 1, 1))
        self.assertEqual(yseq.shape, (2, 2, 1))
        self.assertEqual(ycat.shape, (2, 2, 1))

        self.assertAllclose(xseq, jnp.stack((d.at[0:1,:].get(),
                                             d.at[1:2,:].get())))
        self.assertAllclose(xcat, jnp.stack((c.at[0:1,:].get(),
                                             c.at[1:2,:].get())))
        self.assertAllclose(yseq, jnp.stack((d.at[1:3,:].get(),
                                             d.at[2:4,:].get())))
        self.assertAllclose(ycat, jnp.stack((c.at[1:3,:].get(),
                                             c.at[2:4,:].get())))

    def test_ensure_range(self):
        d = jnp.arange(30, dtype=float)
        seq = SeqData(d, xLen=2, yLen=2, batch_size=2)
        for i in range(seq.nbatch):
            with self.subTest(i=i):
                x, y = seq.ibatch(i)
                self.assertNotAllclose(x, y)
                self.assertAny(y != 0)

        seq.shuffle(jax.random.PRNGKey(30))
        for i in range(seq.nbatch):
            with self.subTest(i=i):
                x, y = seq.ibatch(i)
                self.assertNotAllclose(x, y)
                self.assertAny(y != 0)


    def test_vmap(self):
        d = jnp.reshape(jnp.arange(10, dtype=float), (-1, 1))
        seq = SeqData(d, xLen=1, yLen=1, batch_size=2)

        v = seq.vmap(lambda x, y: (x+y).at[:, 0, 0].get())
        ret = jnp.asarray([[1, 3],
                           [5, 7],
                           [9, 11],
                           [13,15]])
        self.assertEqual(v.shape, ret.shape)
        self.assertAllclose(v, ret)

    def test_scan(self):
        d = jnp.reshape(jnp.arange(10, dtype=float), (-1, 1))
        seq = SeqData(d, xLen=1, yLen=1, batch_size=2)

        v, _ = seq.scan(lambda c, x, y:  (c + (x+y).at[:, 0, 0].get(), None),
                        jnp.zeros(2))
        self.assertEqual(v.shape, (2,))
        self.assertAllclose(v, jnp.asarray([28, 36]))



if __name__ == "__main__":
    unittest.main()
