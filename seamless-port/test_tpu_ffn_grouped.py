import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from aggregate_tpu_ffn_split import main as aggregate_split_main
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


    def test_mixed_fp32_first_row_uses_fixed_32_channel_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def make_meta(variant, workload, rows, group_size):
                groups = 1024 // group_size
                return {
                    "variant": variant,
                    "workload": workload,
                    "status": "CONVERTED",
                    "rows": rows,
                    "group_size": group_size,
                    "output_groups": groups,
                    "output_group_to_tensor_index": list(range(groups)),
                    "output_quantizations": [[1.0, 0] for _ in range(groups)],
                }

            full = make_meta("full", "m32-full", 32, 32)
            first = make_meta("first", "m1-prefill-first", 1, 1024)
            tail = make_meta("tail", "m31-prefill-tail", 31, 64)
            manifest = [full, first, tail]
            (root / "manifest.json").write_text(json.dumps(manifest))

            for item in manifest:
                (root / (item["variant"] + ".source-reference.bin")).write_bytes(
                    np.zeros((item["rows"], 1024), dtype="<f4").tobytes()
                )
                (root / (item["variant"] + "-cpu-output.bin")).write_bytes(
                    np.zeros(item["rows"] * 1024, dtype=np.int8).tobytes()
                )

            first_fp32 = root / "first-fp32-output.bin"
            first_fp32.write_bytes(np.zeros(1024, dtype="<f4").tobytes())
            output_json = root / "mixed-result.json"
            argv = [
                "aggregate_tpu_ffn_split.py",
                str(root / "manifest.json"),
                "--backend", "cpu",
                "--first-group-size", "1024",
                "--tail-group-size", "64",
                "--first-fp32-output", str(first_fp32),
                "--output", str(output_json),
            ]
            with patch.object(sys, "argv", argv):
                aggregate_split_main()

            result = json.loads(output_json.read_text())
            self.assertEqual(result["first_mode"], "fp32_cpu")
            self.assertTrue(result["precision_screen"]["pass"])
            self.assertEqual(
                result["fixed_evaluation_group_size"], 32
            )

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
