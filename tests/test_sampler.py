import torch
import unittest
from specdec.sampler import generate_speculative_tokens, validate_tokens
from specdec.models import load_model, load_tokenizer

class TestSampler(unittest.TestCase):
    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.approx_model = load_model("meta-llama/Llama-2-7b-hf", self.device)
        self.target_model = load_model("meta-llama/Llama-2-70b-hf", self.device)
        self.tokenizer = load_tokenizer("meta-llama/Llama-2-70b-hf")
        self.input_text = "Test speculative decoding"
        self.input_ids = self.tokenizer.encode(self.input_text, return_tensors='pt').to(self.device)

    def test_generate_speculative_tokens(self):
        speculative_tokens = generate_speculative_tokens(self.approx_model, self.input_ids, gamma=4)
        self.assertEqual(speculative_tokens.shape[1], 4, "Generated speculative tokens should match gamma value.")

    def test_validate_tokens(self):
        speculative_tokens = generate_speculative_tokens(self.approx_model, self.input_ids, gamma=4)
        accepted_tokens = validate_tokens(self.target_model, self.input_ids, speculative_tokens)
        self.assertTrue(accepted_tokens.shape[0] <= 4, "Accepted tokens should not exceed gamma value.")

if __name__ == "__main__":
    unittest.main()
