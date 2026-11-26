import io
import struct
import unittest
import wave
from reference_stream import pcm_frames


class PcmFramesTest(unittest.TestCase):
    def test_contiguous_frames_and_single_eof(self):
        stream = io.BytesIO()
        with wave.open(stream, "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(16000)
            out.writeframes(struct.pack("<7h", -32768, 0, 1, 2, 3, 4, 32767))
        stream.seek(0)
        with wave.open(stream, "rb") as source:
            parts = list(pcm_frames(source, 3))
        self.assertEqual([len(x[0]) for x in parts], [3, 3, 1])
        self.assertEqual([x[1] for x in parts], [False, False, True])
        self.assertEqual([round(v * 32768) for samples, _ in parts for v in samples],
                         [-32768, 0, 1, 2, 3, 4, 32767])


if __name__ == "__main__":
    unittest.main()
