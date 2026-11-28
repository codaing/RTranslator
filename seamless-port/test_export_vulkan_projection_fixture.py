import unittest
import numpy as np
from export_vulkan_projection_fixture import pack_fixture, parse_fixture_header


class ProjectionFixtureTest(unittest.TestCase):
    def test_header_and_layout(self):
        x = np.arange(3 * 8, dtype=np.float32).reshape(3, 8)
        w = (np.arange(8 * 12, dtype=np.float32).reshape(8, 12) / 16).astype(np.float16)
        y = x @ w.astype(np.float32)
        blob = pack_fixture(x, w, y)
        self.assertEqual(parse_fixture_header(blob), {"M": 3, "K": 8, "N": 12, "bytes": len(blob)})
        self.assertEqual(blob[:4], b"SVP1")
        self.assertEqual(len(blob), 20 + x.size * 4 + w.size * 2)

    def test_rejects_bad_dimensions(self):
        x = np.ones((2, 7), np.float32)
        w = np.ones((7, 8), np.float16)
        y = np.ones((2, 8), np.float32)
        with self.assertRaises(ValueError):
            pack_fixture(x, w, y)


if __name__ == "__main__":
    unittest.main()
