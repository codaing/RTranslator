#!/usr/bin/env python3
import tempfile
import unittest
from pathlib import Path
from summarize_compact_ffn_repeat import parse_log

SAMPLE = """
Replacing 2 out of 2 node(s) with delegate (TfLiteXNNPackDelegate) node, yielding 1 partitions for subgraph 0.
The input model file size (MB): 4.20184
Explicitly applied XNNPACK delegate, and the model graph will be completely executed by the delegate.
count=3272 first=306 curr=226 min=138 max=13922 avg=302.43 std=343 p5=209 median=263 p95=398
Inference timings in us: Init: 64756, First inference: 4923, Warmup (avg): 336.134, Inference (avg): 302.43
"""


class RepeatParserTest(unittest.TestCase):
    def test_parse_tflite_timing_and_delegate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.log"
            path.write_text(SAMPLE)
            result = parse_log(path)
        self.assertEqual(result["count"], 3272)
        self.assertAlmostEqual(result["reported_avg_us"], 302.43)
        self.assertEqual(result["p95_us"], 398)
        self.assertEqual(result["init_us"], 64756)
        self.assertTrue(result["full_xnnpack"])

    def test_reject_missing_delegate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.log"
            path.write_text(SAMPLE.replace(
                "Replacing 2 out of 2 node(s) with delegate (TfLiteXNNPackDelegate) node, yielding 1 partitions for subgraph 0.",
                "",
            ))
            with self.assertRaises(ValueError):
                parse_log(path)


if __name__ == "__main__":
    unittest.main()
