import unittest
from logging import getLogger
import re

from tsax.random import initPRNGKey
from tsax.testing import TestCase


class TestRandom(TestCase):
    def test_init_with_seed(self):
        seed = 42
        other_seed = 0

        self.assertAllclose(initPRNGKey(seed), initPRNGKey(seed))
        self.assertNotAllclose(initPRNGKey(seed), initPRNGKey(other_seed))

    def test_init_without_seed(self):
        self.assertNotAllclose(initPRNGKey(), initPRNGKey())

    def test_init_logs(self):
        seed = 256

        logger = getLogger("tsax.random")
        with self.assertLogs(logger, level="INFO") as L1:
            initPRNGKey(seed)

        self.assertEqual(len(L1.output), 1)
        self.assertIn(f"Random Seed: {seed:x}", L1.output[0])

        with self.assertLogs(logger, level="INFO") as L2:
            initPRNGKey()

        self.assertEqual(len(L2.output), 1)
        self.assertIsNotNone(re.search(r"Random Seed: [0-9a-f]+", L2.output[0]),
                             msg=L2.output[0])

if __name__ == "__main__":
    unittest.main()
