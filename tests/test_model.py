
import unittest
from specdec.models import load_model, load_tokenizer

class TestModels(unittest.TestCase):
    def test_load_model(self):
        model = load_model("meta-llama/Llama-2-7b-hf", device='cpu')
        self.assertIsNotNone(model, "Model should be loaded successfully")

    def test_load_tokenizer(self):
        tokenizer = load_tokenizer("meta-llama/Llama-2-7b-hf")
        self.assertIsNotNone(tokenizer, "Tokenizer should be loaded successfully")

if __name__ == "__main__":
    unittest.main()
