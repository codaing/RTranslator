import tempfile
import unittest
from pathlib import Path

import numpy as np

from compare_tpu_ffn_outputs import (
    group_error_metrics,
    reconstruct,
    row_error_metrics,
    split_prefill_rows,
)


class GroupedFfnAuditTest(unittest.TestCase):
    def test_split_prefill_rows_preserves_source_order(self):
        x = np.arange(32 * 4, dtype=np.float32).reshape(32, 4)
        y = x + 1000
        (x0, y0), (xt, yt) = split_prefill_rows(x, y)
        self.assertEqual(x0.shape, (1, 4))
        self.assertEqual(xt.shape, (31, 4))
        np.testing.assert_array_equal(np.concatenate([x0, xt]), x)
        np.testing.assert_array_equal(np.concatenate([y0, yt]), y)

    def test_split_rejects_invalid_shapes(self):
        with self.assertRaises(ValueError):
            split_prefill_rows(np.zeros((1, 4)), np.zeros((1, 4)))
        with self.assertRaises(ValueError):
            split_prefill_rows(np.zeros((4, 4)), np.zeros((4, 3)))

    def test_row_metrics_expose_one_bad_row(self):
        reference = np.ones((3, 16), dtype=np.float32)
        actual = reference.copy()
        actual[2] += 0.2
        result = row_error_metrics(reference, actual)
        self.assertEqual(result["worst_row_index"], 2)
        self.assertGreater(result["max_row_nrms"], 0.19)
        self.assertEqual(result["row_nrms"][0], 0.0)

    def test_group_metrics_expose_one_bad_output_group(self):
        reference = np.ones((2, 16), dtype=np.float32)
        actual = reference.copy()
        actual[:, 8:] += 0.1
        result = group_error_metrics(reference, actual, 8)
        self.assertEqual(result["worst_group_index"], 1)
        self.assertEqual(result["worst_group_channels"], [8, 16])
        self.assertEqual(result["groups_over_5pct"], 1)
        self.assertGreater(result["max_group_nrms"], 0.09)

    def test_reconstruct_uses_manifest_tensor_order_and_scales(self):
        raw = np.array(
            [[[1, 2]], [[3, 4]]], dtype=np.int8
        ).tobytes()
        meta = {
            "rows": 1,
            "group_size": 2,
            "output_groups": 2,
            "output_group_to_tensor_index": [1, 0],
            "output_quantizations": [[0.5, 0], [2.0, 1]],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.bin"
            path.write_bytes(raw)
            actual = reconstruct(path, meta)
        np.testing.assert_allclose(actual, [[4.0, 6.0, 0.5, 1.0]])


if __name__ == "__main__":
    unittest.main()
