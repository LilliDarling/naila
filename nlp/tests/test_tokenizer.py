import unittest
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from tokenizer.base_tokenizer import NailaTokenizer

class TestNailaTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = NailaTokenizer()
        cls.test_text = "Don’t you love 🤗 Transformers? We sure do."
        
    def test_process_text(self):
        result = self.tokenizer.process_text(self.test_text)
        self.assertIn('tokens', result)
        self.assertIn('lemmas', result)
        self.assertIn('pos_tags', result)
        self.assertIn('entities', result)
        self.assertIn('noun_chunks', result)
        self.assertIn('dependencies', result)
        
    def test_transformer_encoding(self):
        result = self.tokenizer.get_transformer_encoding(self.test_text)
        self.assertIn('input_ids', result)
        self.assertIn('attention_mask', result)
        self.assertIn('tokens', result)
        self.assertEqual(result['input_ids'].shape[1], 512)
        
    def test_sentence_embedding(self):
        embedding = self.tokenizer.get_sentence_embedding(self.test_text)
        self.assertEqual(len(embedding.shape), 1)
        
    def test_similarity_score(self):
        text2 = "Don’t you love transformers? We sure do."
        similarity = self.tokenizer.similarity_score(self.test_text, text2)
        self.assertGreaterEqual(similarity, -1)
        self.assertLessEqual(similarity, 1)
        
    def test_batch_process(self):
        texts = [
            "The quick brown mouse jumps.",
            "Python is a programming language."
        ]
        results = self.tokenizer.batch_process(texts, include_embeddings=True)
        self.assertEqual(len(results), 2)
        self.assertIn('spacy', results[0])
        self.assertIn('transformer', results[0])
        self.assertIn('embedding', results[0])

if __name__ == '__main__':
    unittest.main()