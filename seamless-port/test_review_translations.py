import unittest
from types import SimpleNamespace
from review_translations import decode_tokens

class TranslationReviewTest(unittest.TestCase):
    def test_decode_and_validate_prefix(self):
        class Prefix:
            def tolist(self):return [2,7]
        class Tokenizer:
            vocab_info=SimpleNamespace(size=8)
            model=SimpleNamespace(index_to_token=lambda i:{4:"▁square",5:"▁root",6:"▁two"}[i])
            def create_encoder(self,**kwargs):return SimpleNamespace(prefix_indices=Prefix())
        tok=Tokenizer()
        self.assertEqual(decode_tokens([2,7,4,5,6],tok),"square root two")
        with self.assertRaises(ValueError):decode_tokens([2,6,4],tok)
        with self.assertRaises(ValueError):decode_tokens([2,7,8],tok)
